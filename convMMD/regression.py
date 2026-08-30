"""Linear errors-in-variables regression with a convMMD latent density."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .core.losses import BandwidthInput, _normalize_bandwidths, mmd_laplace_kernel
from .density_models.nf import NormalizingFlowDensity


@dataclass(frozen=True)
class LinearMeasurementErrorConfig:
    """Effective settings used by :func:`fit_measurement_error_regression`."""

    measurement_error_std: float
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    eval_every: int
    num_blocks: int
    num_bins: int
    hidden_features: int
    tail_bound: float
    max_derivative: float
    minimum_residual_std: float
    seed: int
    device: str


class LinearMeasurementErrorModel(nn.Module):
    """Fitted latent density and scalar linear-regression parameters."""

    def __init__(
        self,
        latent_model: NormalizingFlowDensity,
        *,
        measurement_error_std: float,
        intercept_init: float,
        slope_init: float,
        residual_std_init: float,
        minimum_residual_std: float,
    ):
        super().__init__()
        self.latent_model = latent_model
        self.register_buffer(
            "measurement_error_std",
            torch.tensor(float(measurement_error_std)),
        )
        self.intercept = nn.Parameter(torch.tensor(float(intercept_init)))
        self.slope = nn.Parameter(torch.tensor(float(slope_init)))
        self.minimum_residual_std = float(minimum_residual_std)
        positive_part = max(
            float(residual_std_init) - self.minimum_residual_std,
            1e-6,
        )
        self.raw_residual_std = nn.Parameter(
            torch.tensor(math.log(math.expm1(positive_part)))
        )

    @property
    def residual_std(self) -> torch.Tensor:
        """Strictly positive response-noise standard deviation."""
        return F.softplus(self.raw_residual_std) + self.minimum_residual_std

    def sample_observed(
        self, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw latent covariates, noisy covariates, and responses."""
        latent = self.latent_model.sample(num_samples)
        observed_covariate = latent + self.measurement_error_std * torch.randn_like(
            latent
        )
        response = (
            self.intercept
            + self.slope * latent
            + self.residual_std * torch.randn_like(latent)
        )
        return latent, observed_covariate, response


@dataclass
class LinearMeasurementErrorResult:
    """Fitted regression model, diagnostics, and effective configuration."""

    model: LinearMeasurementErrorModel
    history: Dict[str, Any]
    bandwidths: torch.Tensor
    observed_center: torch.Tensor
    observed_scale: torch.Tensor
    config: LinearMeasurementErrorConfig

    @property
    def intercept(self) -> float:
        return float(self.model.intercept.detach().cpu())

    @property
    def slope(self) -> float:
        return float(self.model.slope.detach().cpu())

    @property
    def residual_std(self) -> float:
        return float(self.model.residual_std.detach().cpu())

    @property
    def latent_model(self) -> NormalizingFlowDensity:
        return self.model.latent_model

    def sample_observed(
        self,
        num_samples: int,
        *,
        seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw reproducible forward samples without changing caller RNG state."""
        num_samples = _validate_integer(num_samples, name="num_samples", minimum=1)
        seed = _validate_seed(seed)
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        try:
            with _fork_seed(seed, device), torch.no_grad():
                return self.model.sample_observed(num_samples)
        finally:
            self.model.train(was_training)


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


def _validate_seed(seed):
    seed = _validate_integer(seed, name="seed", minimum=0)
    if seed > 2**63 - 1:
        raise ValueError("seed must be at most 2**63 - 1")
    return seed


def _validate_vector(value, *, name):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim != 1:
        raise ValueError(f"{name} must have shape (n,) or (n, 1)")
    if value.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two observations")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value.detach()


def _validate_measurement_error_std(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _validate_real(
            value,
            name="measurement_error_std",
            lower_bound=0.0,
            strict=True,
        )
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("measurement_error_std must be a numeric scalar") from error
    if tensor.ndim != 0 or tensor.dtype == torch.bool or tensor.is_complex():
        raise ValueError("measurement_error_std must be a numeric scalar")
    return _validate_real(
        tensor.item(),
        name="measurement_error_std",
        lower_bound=0.0,
        strict=True,
    )


def _resolve_device(device) -> torch.device:
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid device: {device!r}") from error
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("regression currently supports CPU or CUDA devices")
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


def _corrected_moment_initialization(
    observed_covariate: torch.Tensor,
    response: torch.Tensor,
    measurement_error_std: float,
    minimum_residual_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centered_covariate = observed_covariate - observed_covariate.mean()
    centered_response = response - response.mean()
    observed_variance = centered_covariate.square().mean()
    latent_variance = observed_variance - measurement_error_std**2
    if latent_variance.item() <= 0:
        raise ValueError(
            "observed covariate variance must exceed the known "
            "measurement-error variance"
        )

    slope = (centered_covariate * centered_response).mean() / latent_variance
    intercept = response.mean() - slope * observed_covariate.mean()
    residual_variance = response.var(unbiased=False) - slope.square() * latent_variance
    residual_std = torch.sqrt(
        torch.clamp(residual_variance, min=minimum_residual_std**2)
    )
    return intercept, slope, residual_std


def fit_measurement_error_regression(
    observed_covariate: torch.Tensor,
    response: torch.Tensor,
    measurement_error_std,
    *,
    steps: int = 1500,
    batch_size: int = 512,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-5,
    max_grad_norm: float = 1.0,
    bandwidths: Optional[BandwidthInput] = None,
    eval_every: int = 100,
    num_blocks: int = 4,
    num_bins: int = 4,
    hidden_features: int = 32,
    tail_bound: float = 3.0,
    max_derivative: float = 3.0,
    minimum_residual_std: float = 0.02,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> LinearMeasurementErrorResult:
    """Fit the demonstrated scalar linear errors-in-variables model.

    Only the observed noisy covariate, response, and externally known Gaussian
    covariate-error standard deviation are used for fitting. Simulation truth
    is intentionally not accepted by this API.
    """
    observed_covariate = _validate_vector(
        observed_covariate, name="observed_covariate"
    )
    response = _validate_vector(response, name="response")
    if observed_covariate.shape != response.shape:
        raise ValueError("observed_covariate and response must have the same length")

    measurement_error_std = _validate_measurement_error_std(
        measurement_error_std
    )
    steps = _validate_integer(steps, name="steps", minimum=1)
    batch_size = _validate_integer(batch_size, name="batch_size", minimum=2)
    eval_every = _validate_integer(eval_every, name="eval_every", minimum=1)
    num_blocks = _validate_integer(num_blocks, name="num_blocks", minimum=1)
    num_bins = _validate_integer(num_bins, name="num_bins", minimum=2)
    hidden_features = _validate_integer(
        hidden_features, name="hidden_features", minimum=1
    )
    seed = _validate_seed(seed)
    learning_rate = _validate_real(
        learning_rate, name="learning_rate", lower_bound=0.0, strict=True
    )
    weight_decay = _validate_real(
        weight_decay, name="weight_decay", lower_bound=0.0, strict=False
    )
    max_grad_norm = _validate_real(
        max_grad_norm, name="max_grad_norm", lower_bound=0.0, strict=True
    )
    tail_bound = _validate_real(
        tail_bound, name="tail_bound", lower_bound=0.0, strict=True
    )
    max_derivative = _validate_real(
        max_derivative, name="max_derivative", lower_bound=1.0, strict=True
    )
    minimum_residual_std = _validate_real(
        minimum_residual_std,
        name="minimum_residual_std",
        lower_bound=0.0,
        strict=True,
    )
    resolved_device = _resolve_device(device)

    observed_covariate = observed_covariate.to(resolved_device)
    response = response.to(resolved_device)
    observed_pairs = torch.stack((observed_covariate, response), dim=1)
    observed_center = observed_pairs.mean(dim=0, keepdim=True).detach()
    observed_scale = observed_pairs.std(
        dim=0, unbiased=False, keepdim=True
    ).detach()
    if not (observed_scale > 0).all().item():
        raise ValueError("observed_covariate and response must each vary")
    standardized_observed = (observed_pairs - observed_center) / observed_scale

    intercept_init, slope_init, residual_std_init = (
        _corrected_moment_initialization(
            observed_covariate,
            response,
            measurement_error_std,
            minimum_residual_std,
        )
    )
    effective_batch_size = min(batch_size, observed_pairs.shape[0])

    with _fork_seed(seed, resolved_device):
        latent_model = NormalizingFlowDensity(
            dim=1,
            flow_type="nsf",
            num_blocks=num_blocks,
            num_bins=num_bins,
            hidden_features=hidden_features,
            tail_bound=tail_bound,
            max_derivative=max_derivative,
        )
        model = LinearMeasurementErrorModel(
            latent_model,
            measurement_error_std=measurement_error_std,
            intercept_init=intercept_init.item(),
            slope_init=slope_init.item(),
            residual_std_init=residual_std_init.item(),
            minimum_residual_std=minimum_residual_std,
        ).to(device=resolved_device, dtype=torch.float32)

        if bandwidths is None:
            effective_bandwidths = torch.logspace(
                -0.7,
                0.8,
                8,
                device=resolved_device,
                dtype=torch.float32,
            )
        else:
            effective_bandwidths = _normalize_bandwidths(
                bandwidths, observed_pairs
            )

        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        history = {
            "step": [],
            "loss": [],
            "intercept": [],
            "slope": [],
            "residual_std": [],
        }
        progress = tqdm(range(steps), disable=not verbose)
        for step in progress:
            model.train()
            learning_rate_now = learning_rate * 0.5 * (
                1.0 + math.cos(math.pi * step / steps)
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_now

            indices = torch.randint(
                0,
                observed_pairs.shape[0],
                (effective_batch_size,),
                device=resolved_device,
            )
            observed_batch = standardized_observed[indices]
            _, generated_covariate, generated_response = model.sample_observed(
                effective_batch_size
            )
            generated_pairs = torch.cat(
                (generated_covariate, generated_response), dim=1
            )
            standardized_generated = (
                generated_pairs - observed_center
            ) / observed_scale

            optimizer.zero_grad()
            loss = mmd_laplace_kernel(
                standardized_generated,
                observed_batch,
                effective_bandwidths,
            )
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite regression loss at step {step + 1}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )
            optimizer.step()

            should_record = (
                step == 0
                or (step + 1) % eval_every == 0
                or step + 1 == steps
            )
            if should_record:
                history["step"].append(step + 1)
                history["loss"].append(float(loss.detach().cpu()))
                history["intercept"].append(float(model.intercept.detach().cpu()))
                history["slope"].append(float(model.slope.detach().cpu()))
                history["residual_std"].append(
                    float(model.residual_std.detach().cpu())
                )
                progress.set_postfix(
                    loss=f"{history['loss'][-1]:.6f}",
                    slope=f"{history['slope'][-1]:.3f}",
                )
        model.eval()

    config = LinearMeasurementErrorConfig(
        measurement_error_std=measurement_error_std,
        steps=steps,
        batch_size=effective_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        eval_every=eval_every,
        num_blocks=num_blocks,
        num_bins=num_bins,
        hidden_features=hidden_features,
        tail_bound=tail_bound,
        max_derivative=max_derivative,
        minimum_residual_std=minimum_residual_std,
        seed=seed,
        device=str(resolved_device),
    )
    return LinearMeasurementErrorResult(
        model=model,
        history=history,
        bandwidths=effective_bandwidths.detach().cpu(),
        observed_center=observed_center.detach().cpu(),
        observed_scale=observed_scale.detach().cpu(),
        config=config,
    )


__all__ = [
    "LinearMeasurementErrorConfig",
    "LinearMeasurementErrorModel",
    "LinearMeasurementErrorResult",
    "fit_measurement_error_regression",
]
