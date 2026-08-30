"""
MMD loss functions for convMMD training.
"""

import math
from typing import List, Sequence, Union

import torch


BandwidthInput = Union[torch.Tensor, Sequence[float], float]


def _normalize_bandwidths(
    bandwidths: BandwidthInput,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return bandwidths as a validated one-dimensional tensor."""
    normalized = torch.as_tensor(
        bandwidths,
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(-1)

    if normalized.numel() == 0:
        raise ValueError("bandwidths must contain at least one value")
    if not torch.isfinite(normalized).all().item():
        raise ValueError("bandwidths must contain only finite values")
    if not (normalized > 0).all().item():
        raise ValueError("bandwidths must be strictly positive")

    return normalized


def _validate_sample_shapes(x: torch.Tensor, y: torch.Tensor) -> None:
    """Validate the matrix shapes shared by sample-based calculations."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            "x and y must be two-dimensional tensors with shape (n, d)"
        )
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")


def _validate_mmd_samples(x: torch.Tensor, y: torch.Tensor) -> None:
    """Validate the sample shapes required by the unbiased MMD estimator."""
    _validate_sample_shapes(x, y)
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError(
            "The unbiased MMD U-statistic requires at least two samples "
            "from each distribution"
        )


def mmd_laplace_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: BandwidthInput,
) -> torch.Tensor:
    """
    Compute MMD^2 U-statistic with Laplace kernel (ordinary-smooth).

    Laplace kernel: k(x, y) = exp(-||x - y||_1 / sigma)

    Args:
        x: First sample (n, d)
        y: Second sample (m, d)
        bandwidths: Kernel bandwidth(s)

    Returns:
        MMD^2 U-statistic (scalar)
    """
    _validate_mmd_samples(x, y)
    bandwidths = _normalize_bandwidths(bandwidths, x)

    nx = x.shape[0]
    ny = y.shape[0]

    # L1 distances
    # For 1D: |x_i - x_j|
    dist_xx = torch.cdist(x, x, p=1)  # (n, n)
    dist_yy = torch.cdist(y, y, p=1)  # (m, m)
    dist_xy = torch.cdist(x, y, p=1)  # (n, m)

    mmd_sum = 0.0
    for sigma in bandwidths:
        inv_sigma = 1.0 / sigma

        K_xx = torch.exp(-dist_xx * inv_sigma)
        K_yy = torch.exp(-dist_yy * inv_sigma)
        K_xy = torch.exp(-dist_xy * inv_sigma)

        # U-statistic (excludes diagonal) — matches JAX mmd_loss_vstats_laplace
        term_xx = (K_xx.sum() - K_xx.diagonal().sum()) / (nx * (nx - 1))
        term_yy = (K_yy.sum() - K_yy.diagonal().sum()) / (ny * (ny - 1))
        term_xy = 2.0 * K_xy.sum() / (nx * ny)

        mmd_sum = mmd_sum + (term_xx + term_yy - term_xy)

    return mmd_sum / len(bandwidths)


def mmd_gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: BandwidthInput,
) -> torch.Tensor:
    """
    Compute MMD^2 U-statistic with Gaussian (RBF) kernel (super-smooth).

    Gaussian kernel: k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Args:
        x: First sample (n, d)
        y: Second sample (m, d)
        bandwidths: Kernel bandwidth(s)

    Returns:
        MMD^2 U-statistic (scalar)
    """
    _validate_mmd_samples(x, y)
    bandwidths = _normalize_bandwidths(bandwidths, x)

    nx = x.shape[0]
    ny = y.shape[0]

    # Squared L2 distances
    xx = (x * x).sum(dim=1, keepdim=True)  # (n, 1)
    yy = (y * y).sum(dim=1, keepdim=True)  # (m, 1)

    dist_sq_xx = xx + xx.t() - 2.0 * x @ x.t()  # (n, n)
    dist_sq_yy = yy + yy.t() - 2.0 * y @ y.t()  # (m, m)
    dist_sq_xy = xx + yy.t() - 2.0 * x @ y.t()  # (n, m)

    # Clamp to avoid numerical issues
    dist_sq_xx = dist_sq_xx.clamp(min=0.0)
    dist_sq_yy = dist_sq_yy.clamp(min=0.0)
    dist_sq_xy = dist_sq_xy.clamp(min=0.0)

    mmd_sum = 0.0
    for sigma in bandwidths:
        gamma = 1.0 / (2.0 * sigma * sigma)

        K_xx = torch.exp(-dist_sq_xx * gamma)
        K_yy = torch.exp(-dist_sq_yy * gamma)
        K_xy = torch.exp(-dist_sq_xy * gamma)

        # U-statistic (excludes diagonal)
        diag_xx = torch.diag(K_xx)
        diag_yy = torch.diag(K_yy)

        term_xx = (K_xx.sum() - diag_xx.sum()) / (nx * (nx - 1))
        term_yy = (K_yy.sum() - diag_yy.sum()) / (ny * (ny - 1))
        term_xy = 2.0 * K_xy.sum() / (nx * ny)

        mmd_sum = mmd_sum + (term_xx + term_yy - term_xy)

    return mmd_sum / len(bandwidths)


def compute_bandwidth_median_heuristic(
    x: torch.Tensor,
    y: torch.Tensor,
    quantiles: List[float] = None,
) -> torch.Tensor:
    """
    Compute kernel bandwidths using median heuristic with multiple quantiles.

    Args:
        x: First sample (n, d)
        y: Second sample (m, d)
        quantiles: Quantiles to use (default: [0.1, 0.2, ..., 0.9])

    Returns:
        Bandwidth values
    """
    _validate_sample_shapes(x, y)
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("x and y must each contain at least one sample")

    if quantiles is None:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    z = torch.cat([x, y], dim=0)
    n = z.shape[0]

    # Pairwise L2 distances (not squared)
    dists = torch.cdist(z, z, p=2)

    # Get upper triangle (exclude diagonal and duplicates)
    mask = torch.triu(torch.ones(n, n, device=z.device), diagonal=1).bool()
    dists_flat = dists[mask]

    # Filter out very small distances
    dists_filtered = dists_flat[dists_flat > 1e-5]
    if dists_filtered.numel() == 0:
        raise ValueError(
            "Cannot compute bandwidths because no pairwise distance exceeds 1e-5"
        )

    quantile_tensor = torch.as_tensor(
        quantiles,
        device=z.device,
        dtype=z.dtype,
    ).reshape(-1)
    if quantile_tensor.numel() == 0:
        raise ValueError("quantiles must contain at least one value")
    if not torch.isfinite(quantile_tensor).all().item():
        raise ValueError("quantiles must contain only finite values")
    if not ((quantile_tensor >= 0) & (quantile_tensor <= 1)).all().item():
        raise ValueError("quantiles must lie between 0 and 1")

    # Distances are already unsquared, so their quantiles are bandwidth values.
    quantile_vals = torch.quantile(dists_filtered, quantile_tensor)

    # Match JAX heuristic: divide by log10(2n) * sqrt(d), n = data count, d = dim
    n_data = x.shape[0]
    d = x.shape[1]
    factor = 1.0 / (math.log10(2 * n_data) * math.sqrt(d))
    return quantile_vals * factor
