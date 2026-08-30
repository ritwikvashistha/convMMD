"""Normalizing flow models using the ``nflows`` library.

The spline calculation is adapted from ``nflows==0.14`` under its MIT License;
see ``THIRD_PARTY_NOTICES.md``. convMMD uses a package-local autoregressive
transform that caps spline derivatives to match the bounded-derivative method.
Importing this module does not modify ``nflows`` or unrelated models in the
Python process.

In v0.1.0, ``create_nsf_1d`` retains a standard-normal base distribution.
Its ``base_mean`` and ``base_std`` arguments are stored as metadata but do not
change that base. This limitation is preserved here to avoid silently changing
the scientific method for the initial research-preview release.
"""

import copy

import torch
import torch.nn as nn
import numpy as np

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.distributions.base import Distribution
from nflows.transforms import CompositeTransform
from nflows.transforms.base import Transform, InverseTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform as _NFlowsRQSAutoregressiveTransform,
)
from nflows.transforms.normalization import BatchNorm, ActNorm
from nflows.transforms.permutations import RandomPermutation

import torch.nn.functional as F
from nflows.transforms.base import InputOutsideDomain
from nflows.utils import torchutils


# ============================================================
# Bounded-derivative rational quadratic spline (JAX parity)
#
# nflows parameterizes spline derivatives with an *unbounded* softplus, allowing
# arbitrarily sharp local density spikes. The JAX flow instead caps derivatives
# with a sigmoid cap to keep the density smooth / finite Sobolev norm. We
# reimplement the two spline functions with that single change and call them
# from a package-local nflows-compatible transform.
# ============================================================

MAX_DERIVATIVE = 10.0


def _bounded_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    max_derivative=MAX_DERIVATIVE,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # CHANGED vs nflows: bounded sigmoid derivative instead of unbounded
    # softplus. The cap belongs to the calling transform instance.
    derivatives = min_derivative + (max_derivative - min_derivative) * torch.sigmoid(
        unnormalized_derivatives
    )

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = torchutils.searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = torchutils.searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        # JAX parity: clamp inside sqrt instead of asserting, so a transient bad
        # spline config during sampling degrades gracefully rather than crashing.
        root = (2 * c) / (-b - torch.sqrt(torch.clamp(discriminant, min=1e-14)))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet


def _bounded_unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    max_derivative=MAX_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        # CHANGED vs nflows: logit boundary constant so the bounded-sigmoid param
        # maps a 0 logit's neighbours to derivative 1.0 on the linear tails.
        _p = (1.0 - min_derivative) / (max_derivative - min_derivative)
        constant = np.log(_p / (1.0 - _p))
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        (
            outputs[inside_interval_mask],
            logabsdet[inside_interval_mask],
        ) = _bounded_rational_quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            max_derivative=max_derivative,
        )

    return outputs, logabsdet


class BoundedPiecewiseRationalQuadraticAutoregressiveTransform(
    _NFlowsRQSAutoregressiveTransform
):
    """nflows-compatible RQS transform with an instance-local derivative cap.

    The learned parameter structure is identical to the upstream nflows 0.14
    transform, so v0.1.0 state dictionaries retain the same keys.
    """

    def __init__(self, *args, max_derivative=MAX_DERIVATIVE, **kwargs):
        try:
            max_derivative = float(max_derivative)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "max_derivative must be finite and greater than 1"
            ) from error

        if not np.isfinite(max_derivative) or max_derivative <= 1.0:
            raise ValueError("max_derivative must be finite and greater than 1")

        self.max_derivative = max_derivative
        super().__init__(*args, **kwargs)

        if self.min_derivative >= self.max_derivative:
            raise ValueError("min_derivative must be less than max_derivative")
        if self.tails == "linear" and self.min_derivative >= 1.0:
            raise ValueError("min_derivative must be less than 1 for linear tails")

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]
        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[
            ..., self.num_bins : 2 * self.num_bins
        ]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            scale = np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_widths = unnormalized_widths / scale
            unnormalized_heights = unnormalized_heights / scale

        if self.tails is None:
            spline_fn = _bounded_rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = _bounded_unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError(f"Unsupported tails setting: {self.tails}")

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            max_derivative=self.max_derivative,
            **spline_kwargs,
        )

        return outputs, torchutils.sum_except_batch(logabsdet)


class DiagNormal(Distribution):
    """Diagonal Gaussian base with fixed (non-trainable) mean and std.

    Mirrors the JAX flow's data-matched base z ~ N(data_mean, data_std), so the
    spline transforms only have to model the *shape* of the target rather than
    its location and scale.
    """

    def __init__(self, shape, mean, std):
        super().__init__()
        self._shape = torch.Size(shape)

        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)
        mean_t = torch.full(self._shape, float(mean_t)) if mean_t.numel() == 1 else mean_t.reshape(self._shape)
        std_t = torch.full(self._shape, float(std_t)) if std_t.numel() == 1 else std_t.reshape(self._shape)

        self.register_buffer("loc", mean_t)
        self.register_buffer("scale", std_t)
        self.register_buffer(
            "_log_z",
            torch.tensor(0.5 * int(np.prod(self._shape)) * np.log(2 * np.pi), dtype=torch.float32),
        )

    def _log_prob(self, inputs, context=None):
        z = (inputs - self.loc) / self.scale
        neg_energy = -0.5 * z.pow(2).flatten(start_dim=1).sum(dim=1)
        return neg_energy - torch.log(self.scale).sum() - self._log_z

    def _sample(self, num_samples, context=None):
        eps = torch.randn(num_samples, *self._shape, device=self.loc.device)
        return self.loc + self.scale * eps

    def _mean(self, context=None):
        return self.loc


class AffineCouplingTransform(Transform):
    """Simple affine transform: y = x * exp(log_scale) + shift"""

    def __init__(self, dim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, inputs, context=None):
        outputs = inputs * torch.exp(self.log_scale) + self.shift
        logabsdet = self.log_scale.sum().expand(inputs.shape[0])
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs = (inputs - self.shift) * torch.exp(-self.log_scale)
        logabsdet = -self.log_scale.sum().expand(inputs.shape[0])
        return outputs, logabsdet


def create_nsf_1d(
    num_blocks: int = 4,
    num_bins: int = 16,
    hidden_features: int = 32,
    tail_bound: float = 30.0,
    base_mean: float = 0.0,
    base_std: float = 1.0,
    max_derivative: float = MAX_DERIVATIVE,
) -> Flow:
    """
    Create a 1D Neural Spline Flow for density estimation.

    Uses rational quadratic splines (RQS) similar to the JAX implementation.

    Args:
        num_blocks: Number of flow blocks
        num_bins: Number of bins for spline
        hidden_features: Hidden layer size
        tail_bound: Bound for linear tails
        base_mean: Retained as metadata; the operative base remains standard normal
        base_std: Retained as metadata; the operative base remains standard normal
        max_derivative: Upper bound for spline derivatives

    Returns:
        nflows Flow object
    """
    transforms = []

    for _ in range(num_blocks):
        transforms.append(
            BoundedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=1,
                hidden_features=hidden_features,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                num_blocks=2,
                use_residual_blocks=False,
                activation=torch.relu,
                max_derivative=max_derivative,
            )
        )
        transforms.append(AffineCouplingTransform(dim=1))

    transform = CompositeTransform(transforms)
    base_distribution = StandardNormal([1])

    flow = Flow(transform, base_distribution)
    flow.base_mean = base_mean
    flow.base_std = base_std

    return flow


def create_nsf_nd(
    dim: int,
    num_blocks: int = 4,
    num_bins: int = 8,
    hidden_features: int = 64,
    tail_bound: float = 3.0,
    base_mean=None,
    base_std=None,
    max_derivative: float = MAX_DERIVATIVE,
) -> Flow:
    """
    Create an N-dimensional Neural Spline Flow.

    Args:
        dim: Input dimensionality
        num_blocks: Number of flow blocks
        num_bins: Number of bins for spline
        hidden_features: Hidden layer size
        tail_bound: Bound for linear tails
        base_mean: Optional mean for a fixed diagonal Gaussian base
        base_std: Optional standard deviation for a fixed diagonal Gaussian base
        max_derivative: Upper bound for spline derivatives

    Returns:
        nflows Flow object
    """
    transforms = []

    for i in range(num_blocks):
        transforms.append(
            BoundedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_features,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                num_blocks=2,
                use_residual_blocks=True,
                activation=torch.relu,
                min_bin_width=1e-2,
                min_bin_height=1e-2,
                min_derivative=1e-2,
                max_derivative=max_derivative,
            )
        )
        if i < num_blocks - 1:
            transforms.append(RandomPermutation(features=dim))

    transform = CompositeTransform(transforms)
    if base_mean is not None and base_std is not None:
        base_distribution = DiagNormal([dim], base_mean, base_std)
    else:
        base_distribution = StandardNormal([dim])

    return Flow(transform, base_distribution)


def create_iaf(
    dim: int,
    hidden_features: int = 128,
    num_layers: int = 4,
    num_blocks_per_layer: int = 2,
    dropout_prob: float = 0.0,
    use_residual_blocks: bool = True,
    use_random_permutations: bool = True,
    use_random_masks: bool = False,
    batch_norm_within_layers: bool = False,
    batch_norm_between_layers: bool = False,
    use_actnorm: bool = True,
) -> Flow:
    """
    Create an Inverse Autoregressive Flow (IAF).

    IAF is efficient for sampling (parallel) but slow for density evaluation.
    Good for generative modeling where sampling is the primary operation.

    Args:
        dim: Input dimensionality
        hidden_features: Hidden layer size in MADE networks
        num_layers: Number of IAF layers
        num_blocks_per_layer: Number of residual blocks per MADE
        dropout_prob: Dropout probability
        use_residual_blocks: Use residual connections in MADE
        use_random_permutations: Randomly permute between layers
        use_random_masks: Use random masks in MADE
        batch_norm_within_layers: BatchNorm within MADE
        batch_norm_between_layers: BatchNorm between IAF layers
        use_actnorm: Use ActNorm for data-dependent initialization

    Returns:
        nflows Flow object
    """
    transforms_list = []

    for i in range(num_layers):
        # ActNorm before each IAF block for data-dependent initialization
        if use_actnorm:
            transforms_list.append(ActNorm(dim))

        transforms_list.append(
            InverseTransform(
                MaskedAffineAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    dropout_probability=dropout_prob,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
        )
        if i < num_layers - 1:
            if batch_norm_between_layers:
                transforms_list.append(InverseTransform(BatchNorm(dim)))
            if use_random_permutations:
                transforms_list.append(RandomPermutation(dim))

    return Flow(
        transform=CompositeTransform(transforms_list),
        distribution=StandardNormal([dim]),
    )


def _to_plain_config_value(value):
    """Convert supported construction values to checkpoint-safe primitives."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_plain_config_value(item) for item in value]
    raise TypeError(
        "NormalizingFlowDensity checkpoint configuration values must be "
        "plain scalars, sequences, NumPy arrays, or tensors"
    )


class NormalizingFlowDensity(nn.Module):
    """
    Wrapper for normalizing flow that provides convenient sampling and density evaluation.
    """

    def __init__(
        self,
        dim: int = 1,
        flow_type: str = "nsf",
        num_blocks: int = 4,
        num_bins: int = 16,
        hidden_features: int = 32,
        tail_bound: float = 30.0,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        max_derivative: float = MAX_DERIVATIVE,
        **kwargs,
    ):
        """
        Args:
            dim: Input dimensionality
            flow_type: Type of flow ("nsf" for Neural Spline Flow, "iaf" for IAF)
            num_blocks: Number of flow blocks/layers
            num_bins: Number of bins for spline (NSF only)
            hidden_features: Hidden layer size
            tail_bound: Bound for linear tails (NSF only)
            data_mean: Base mean for multidimensional NSF; metadata only for
                1D NSF and IAF models in v0.1.0
            data_std: Base standard deviation for multidimensional NSF;
                metadata only for 1D NSF and IAF models in v0.1.0
            max_derivative: Upper bound for spline derivatives (NSF only)
            **kwargs: Additional arguments passed to flow constructor
        """
        super().__init__()
        self.dim = dim
        self.flow_type = flow_type
        self.data_mean = data_mean
        self.data_std = data_std
        self.max_derivative = max_derivative
        iaf_kwargs = {}

        if flow_type == "nsf":
            if dim == 1:
                self.flow = create_nsf_1d(
                    num_blocks=num_blocks,
                    num_bins=num_bins,
                    hidden_features=hidden_features,
                    tail_bound=tail_bound,
                    base_mean=data_mean,
                    base_std=data_std,
                    max_derivative=max_derivative,
                )
            else:
                self.flow = create_nsf_nd(
                    dim=dim,
                    num_blocks=num_blocks,
                    num_bins=num_bins,
                    hidden_features=hidden_features,
                    tail_bound=tail_bound,
                    base_mean=data_mean,
                    base_std=data_std,
                    max_derivative=max_derivative,
                )
        elif flow_type == "iaf":
            iaf_kwargs = {
                "num_blocks_per_layer": kwargs.pop("num_blocks_per_layer", 2),
                "dropout_prob": kwargs.pop("dropout_prob", 0.0),
                "use_residual_blocks": kwargs.pop("use_residual_blocks", True),
                "use_random_permutations": kwargs.pop(
                    "use_random_permutations", True
                ),
                "use_random_masks": kwargs.pop("use_random_masks", False),
                "batch_norm_within_layers": kwargs.pop(
                    "batch_norm_within_layers", False
                ),
                "batch_norm_between_layers": kwargs.pop(
                    "batch_norm_between_layers", False
                ),
                "use_actnorm": kwargs.pop("use_actnorm", True),
            }
            self.flow = create_iaf(
                dim=dim,
                hidden_features=hidden_features,
                num_layers=num_blocks,
                **iaf_kwargs,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

        self._checkpoint_model_config = {
            "dim": int(dim),
            "flow_type": str(flow_type),
            "num_blocks": int(num_blocks),
            "num_bins": int(num_bins),
            "hidden_features": int(hidden_features),
            "tail_bound": float(tail_bound),
            "data_mean": _to_plain_config_value(data_mean),
            "data_std": _to_plain_config_value(data_std),
            "max_derivative": float(max_derivative),
            "iaf_kwargs": {
                name: _to_plain_config_value(value)
                for name, value in iaf_kwargs.items()
            },
        }

    def _get_checkpoint_model_config(self):
        """Return a detached plain-Python snapshot of the construction config."""
        return copy.deepcopy(self._checkpoint_model_config)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate log probability of samples."""
        return self.flow.log_prob(x)

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from the flow."""
        return self.flow.sample(n_samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns log probability."""
        return self.log_prob(x)
