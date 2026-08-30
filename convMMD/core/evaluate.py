"""
Evaluation metrics for deconvolution experiments.
"""

import torch
import numpy as np
from typing import Callable, Optional
from scipy import stats


def compute_ise(
    model: torch.nn.Module,
    true_density_fn: Callable,
    device: str = "cpu",
    lower: float = -8.0,
    upper: float = 10.0,
    resolution: int = 2000,
) -> float:
    """
    Compute Integrated Squared Error between model density and true density (1D).

    Args:
        model: Density model with log_prob method
        true_density_fn: Function that evaluates true density
        device: Device for computation
        lower: Lower bound of integration
        upper: Upper bound of integration
        resolution: Number of grid points

    Returns:
        ISE value
    """
    x_grid = np.linspace(lower, upper, resolution)
    dx = (upper - lower) / (resolution - 1)

    true_pdf = true_density_fn(x_grid)

    x_tensor = torch.tensor(x_grid, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        log_probs = model.log_prob(x_tensor)
        est_pdf = torch.exp(log_probs).cpu().numpy()

    sq_diff = (est_pdf - true_pdf) ** 2
    ise = np.sum(sq_diff) * dx

    return float(ise)


def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 1000,
    p: int = 1,
    seed: Optional[int] = None,
) -> float:
    """
    Compute Sliced Wasserstein Distance between two point clouds.

    SWD approximates the Wasserstein distance by averaging 1D Wasserstein
    distances over random projections.

    Args:
        x: First point cloud (n, d)
        y: Second point cloud (m, d)
        n_projections: Number of random projections
        p: Order of Wasserstein distance (1 or 2)
        seed: Random seed for reproducibility. An explicit seed does not change
            the caller's PyTorch random state.

    Returns:
        Sliced Wasserstein distance
    """
    if p not in (1, 2):
        raise ValueError(f"p must be 1 or 2, got {p}")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    x = x.detach().cpu()
    y = y.detach().cpu()

    d = x.shape[1]

    # Generate random directions on unit sphere
    directions = torch.randn(
        n_projections,
        d,
        dtype=x.dtype,
        generator=generator,
    )
    directions = directions / directions.norm(dim=1, keepdim=True)

    # Project samples
    x_proj = x @ directions.T  # (n, n_projections)
    y_proj = y @ directions.T  # (m, n_projections)

    # Sort projections
    x_sorted = torch.sort(x_proj, dim=0)[0]
    y_sorted = torch.sort(y_proj, dim=0)[0]

    # Compute each projected empirical Wasserstein distance. Equal sample counts
    # admit the usual sorted-sample fast path. For unequal counts, integrate the
    # two empirical quantile functions exactly over their combined breakpoints.
    n, m = x_sorted.shape[0], y_sorted.shape[0]
    if n == m:
        differences = torch.abs(x_sorted - y_sorted)
        if p == 1:
            distances = differences.mean(dim=0)
        else:
            distances = differences.square().mean(dim=0).sqrt()
    else:
        x_breakpoints = torch.arange(n + 1, dtype=torch.float64) / n
        y_breakpoints = torch.arange(m + 1, dtype=torch.float64) / m
        breakpoints = torch.unique(
            torch.cat([x_breakpoints, y_breakpoints]), sorted=True
        )
        interval_weights = breakpoints[1:] - breakpoints[:-1]
        interval_midpoints = (breakpoints[1:] + breakpoints[:-1]) / 2

        x_indices = torch.floor(interval_midpoints * n).long().clamp(max=n - 1)
        y_indices = torch.floor(interval_midpoints * m).long().clamp(max=m - 1)
        differences = torch.abs(x_sorted[x_indices] - y_sorted[y_indices])
        weights = interval_weights.to(dtype=x_sorted.dtype).unsqueeze(1)

        if p == 1:
            distances = (differences * weights).sum(dim=0)
        else:
            distances = (differences.square() * weights).sum(dim=0).sqrt()

    return float(distances.mean())


def compute_swd_scaled(
    samples: torch.Tensor,
    reference: torch.Tensor,
    n_projections: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """
    Compute Sliced Wasserstein Distance scaled by sqrt(dimension).

    This scaling makes SWD comparable across dimensions.

    Args:
        samples: Generated samples (n, d)
        reference: Reference samples from true distribution (m, d)
        n_projections: Number of random projections
        seed: Random seed

    Returns:
        SWD * sqrt(d)
    """
    d = samples.shape[1]
    swd = sliced_wasserstein_distance(samples, reference, n_projections, seed=seed)
    return swd * np.sqrt(d)


def compute_mse(
    denoised: torch.Tensor,
    true_latent: torch.Tensor,
) -> float:
    """
    Compute Mean Squared Error for denoising.

    Args:
        denoised: Denoised estimates (n, d)
        true_latent: True latent values (n, d)

    Returns:
        MSE value
    """
    return float(((denoised - true_latent) ** 2).mean())


def compute_w1_exact(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> float:
    """
    Compute exact Wasserstein-1 distance using optimal transport.

    Requires the POT library.

    Args:
        samples_a: First sample (n, d)
        samples_b: Second sample (m, d)

    Returns:
        Exact W1 distance
    """
    try:
        import ot
    except ImportError:
        raise ImportError("POT library required. Install with: pip install POT")

    a_np = np.asarray(samples_a)
    b_np = np.asarray(samples_b)

    n = a_np.shape[0]
    m = b_np.shape[0]
    u = np.ones((n,)) / n
    v = np.ones((m,)) / m

    M = ot.dist(a_np, b_np, metric='euclidean')
    exact_w1 = ot.emd2(u, v, M)

    return float(exact_w1)


def estimate_density_kde(
    samples: np.ndarray,
    eval_points: np.ndarray,
    bw_method: str = 'silverman',
) -> np.ndarray:
    """
    Estimate density using Kernel Density Estimation.

    Args:
        samples: Training samples (n, d)
        eval_points: Points to evaluate density at (m, d)
        bw_method: Bandwidth selection method

    Returns:
        Density estimates at eval_points
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if eval_points.ndim == 1:
        eval_points = eval_points.reshape(-1, 1)

    kde = stats.gaussian_kde(samples.T, bw_method=bw_method)
    return kde(eval_points.T)
