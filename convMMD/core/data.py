"""
Data generation for deconvolution experiments.
"""

import torch
import numpy as np
from typing import Tuple


def _seeded_cpu_generator(seed):
    if seed is None:
        return None
    return torch.Generator(device="cpu").manual_seed(seed)


def generate_1d_laplace_mixture(
    n_samples: int,
    noise_type: str = "laplace",
    noise_std_range: Tuple[float, float] = (0.5, 1.0),
    seed: int = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate data from a 1D Laplace mixture with heteroscedastic noise.

    True latent density: 0.6 * Laplace(3.0, 0.8) + 0.4 * Laplace(-2.0, 1.0)
    where Laplace(loc, std) has scale = std / sqrt(2).

    Args:
        n_samples: Number of samples to generate
        noise_type: Type of noise ("laplace" for ordinary-smooth, "gaussian" for super-smooth)
        noise_std_range: (min_std, max_std) for heteroscedastic noise
        seed: Random seed for reproducibility. An explicit seed does not change
            the caller's PyTorch or NumPy random state.
        device: Device to place tensors on

    Returns:
        theta: True latent samples (n_samples, 1)
        x_noisy: Noisy observations (n_samples, 1)
        noise_std: Per-sample noise standard deviations (n_samples, 1)
    """
    generator = _seeded_cpu_generator(seed)

    # Mixture parameters
    mu1, std1, w1 = 3.0, 0.8, 0.6
    mu2, std2 = -2.0, 1.0

    # Laplace scale = std / sqrt(2)
    scale1 = std1 / np.sqrt(2.0)
    scale2 = std2 / np.sqrt(2.0)

    # Sample component assignments
    component = torch.bernoulli(
        torch.full((n_samples,), w1), generator=generator
    )

    # Sample from Laplace distributions
    samples_1_uniform = torch.rand(n_samples, generator=generator)
    samples_1_log_uniform = torch.rand(n_samples, generator=generator)
    samples_1 = mu1 + scale1 * torch.sign(samples_1_uniform - 0.5) * torch.log(
        1 - 2 * torch.abs(samples_1_log_uniform - 0.5) + 1e-10
    )
    samples_2_uniform = torch.rand(n_samples, generator=generator)
    samples_2_log_uniform = torch.rand(n_samples, generator=generator)
    samples_2 = mu2 + scale2 * torch.sign(samples_2_uniform - 0.5) * torch.log(
        1 - 2 * torch.abs(samples_2_log_uniform - 0.5) + 1e-10
    )

    theta = torch.where(component.bool(), samples_1, samples_2).unsqueeze(1)

    # Heteroscedastic noise standard deviations
    min_std, max_std = noise_std_range
    noise_std = (
        torch.rand(n_samples, 1, generator=generator) * (max_std - min_std)
        + min_std
    )

    # Generate noise
    if noise_type == "laplace":
        noise_scale = noise_std / np.sqrt(2.0)
        noise_uniform = torch.rand(n_samples, 1, generator=generator)
        noise_log_uniform = torch.rand(n_samples, 1, generator=generator)
        noise = torch.sign(noise_uniform - 0.5) * torch.log(
            1 - 2 * torch.abs(noise_log_uniform - 0.5) + 1e-10
        ) * noise_scale
    elif noise_type == "gaussian":
        noise = torch.randn(n_samples, 1, generator=generator) * noise_std
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    x_noisy = theta + noise

    return theta.to(device), x_noisy.to(device), noise_std.to(device)


def true_density_1d_laplace_mixture(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the true latent density (Laplace mixture) at points x.

    Args:
        x: Points at which to evaluate density

    Returns:
        Density values at x
    """
    from scipy import stats

    mu1, std1, w1 = 3.0, 0.8, 0.6
    mu2, std2, w2 = -2.0, 1.0, 0.4

    scale1 = std1 / np.sqrt(2.0)
    scale2 = std2 / np.sqrt(2.0)

    pdf1 = stats.laplace.pdf(x, loc=mu1, scale=scale1)
    pdf2 = stats.laplace.pdf(x, loc=mu2, scale=scale2)

    return w1 * pdf1 + w2 * pdf2


# =============================================================================
# 2D Dataset Generators
# =============================================================================

def generate_moons(
    n_samples: int = 2000,
    noise_std_range: Tuple[float, float] = (0.2, 0.6),
    outlier_fraction: float = 0.03,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate two moons dataset with heteroscedastic Gaussian noise and outliers.

    Args:
        n_samples: Number of samples
        noise_std_range: (min_std, max_std) for heteroscedastic noise
        outlier_fraction: Fraction of samples to be outliers
        seed: Random seed. An explicit seed does not change caller RNG state.
        device: Device to place tensors on

    Returns:
        theta: True latent samples (n_samples, 2)
        x_noisy: Noisy observations (n_samples, 2)
        noise_std: Per-sample noise standard deviations (n_samples, 1)
    """
    from sklearn.datasets import make_moons

    generator = _seeded_cpu_generator(seed)

    # Generate clean moons
    clean_np, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    clean = torch.tensor(clean_np, dtype=torch.float32)

    # Add outliers
    n_outliers = int(n_samples * outlier_fraction)
    if n_outliers > 0:
        outlier_indices = torch.randperm(n_samples, generator=generator)[:n_outliers]
        offset = torch.rand(n_outliers, 2, generator=generator) * 6.0 - 3.0
        clean[outlier_indices] = clean[outlier_indices] + offset

    # Heteroscedastic noise
    min_std, max_std = noise_std_range
    noise_std = (
        torch.rand(n_samples, 1, generator=generator) * (max_std - min_std)
        + min_std
    )
    noise = torch.randn(n_samples, 2, generator=generator) * noise_std

    x_noisy = clean + noise

    # Return original clean (without outliers) for evaluation
    clean_original, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    theta = torch.tensor(clean_original, dtype=torch.float32)

    return theta.to(device), x_noisy.to(device), noise_std.to(device)


def generate_circles(
    n_samples: int = 2000,
    noise_std_range: Tuple[float, float] = (0.5, 1.0),
    outlier_fraction: float = 0.03,
    radius: float = 2.0,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate circle dataset with heteroscedastic Gaussian noise and outliers.

    Args:
        n_samples: Number of samples
        noise_std_range: (min_std, max_std) for heteroscedastic noise
        outlier_fraction: Fraction of samples to be outliers
        radius: Radius of the circle
        seed: Random seed. An explicit seed does not change caller RNG state.
        device: Device to place tensors on

    Returns:
        theta: True latent samples (n_samples, 2)
        x_noisy: Noisy observations (n_samples, 2)
        noise_std: Per-sample noise standard deviations (n_samples, 1)
    """
    generator = _seeded_cpu_generator(seed)

    # Generate angles uniformly
    angles = torch.rand(n_samples, generator=generator) * 2 * np.pi
    clean = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)

    # Store original clean for return
    theta = clean.clone()

    # Add outliers (push to larger radii, but not too extreme)
    n_outliers = int(n_samples * outlier_fraction)
    if n_outliers > 0:
        outlier_indices = torch.randperm(n_samples, generator=generator)[:n_outliers]
        # Outliers at radius 3-5 (more moderate than original 5-10)
        outlier_radii = 3.0 + torch.rand(n_outliers, generator=generator) * 2.0
        clean[outlier_indices, 0] = outlier_radii * torch.cos(angles[outlier_indices])
        clean[outlier_indices, 1] = outlier_radii * torch.sin(angles[outlier_indices])

    # Heteroscedastic noise
    min_std, max_std = noise_std_range
    noise_std = (
        torch.rand(n_samples, 1, generator=generator) * (max_std - min_std)
        + min_std
    )
    noise = torch.randn(n_samples, 2, generator=generator) * noise_std

    x_noisy = clean + noise

    return theta.to(device), x_noisy.to(device), noise_std.to(device)


def generate_checkerboard(
    n_samples: int = 2000,
    noise_std_range: Tuple[float, float] = (0.2, 0.6),
    outlier_fraction: float = 0.03,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate checkerboard dataset with heteroscedastic Gaussian noise and outliers.

    The checkerboard has 8 squares (4x4 grid with alternating pattern).

    Args:
        n_samples: Number of samples
        noise_std_range: (min_std, max_std) for heteroscedastic noise
        outlier_fraction: Fraction of samples to be outliers
        seed: Random seed. An explicit seed does not change caller RNG state.
        device: Device to place tensors on

    Returns:
        theta: True latent samples (n_samples, 2)
        x_noisy: Noisy observations (n_samples, 2)
        noise_std: Per-sample noise standard deviations (n_samples, 1)
    """
    generator = _seeded_cpu_generator(seed)

    # Generate checkerboard pattern
    x1_base = torch.randint(0, 4, (n_samples,), generator=generator)
    x2_base = torch.randint(0, 4, (n_samples,), generator=generator)

    # Ensure checkerboard pattern (alternate squares)
    parity = (x1_base + x2_base) % 2
    x2_base = (x2_base + parity) % 4

    # Add uniform noise within each square
    x1 = x1_base.float() + torch.rand(n_samples, generator=generator)
    x2 = x2_base.float() + torch.rand(n_samples, generator=generator)

    # Center at origin
    clean = torch.stack([x1, x2], dim=1) - 2.0
    theta = clean.clone()

    # Add outliers
    n_outliers = int(n_samples * outlier_fraction)
    if n_outliers > 0:
        outlier_indices = torch.randperm(n_samples, generator=generator)[:n_outliers]
        offset = torch.rand(n_outliers, 2, generator=generator) * 8.0 - 4.0
        clean[outlier_indices] = clean[outlier_indices] + offset

    # Heteroscedastic noise
    min_std, max_std = noise_std_range
    noise_std = (
        torch.rand(n_samples, 1, generator=generator) * (max_std - min_std)
        + min_std
    )
    noise = torch.randn(n_samples, 2, generator=generator) * noise_std

    x_noisy = clean + noise

    return theta.to(device), x_noisy.to(device), noise_std.to(device)


# Dataset registry for easy access
DATASET_REGISTRY = {
    "moons": {
        "fn": generate_moons,
        "default_kwargs": {"noise_std_range": (0.2, 0.6), "outlier_fraction": 0.03},
    },
    "circles": {
        "fn": generate_circles,
        "default_kwargs": {"noise_std_range": (0.5, 1.0), "outlier_fraction": 0.03},
    },
    "checkerboard": {
        "fn": generate_checkerboard,
        "default_kwargs": {"noise_std_range": (0.2, 0.6), "outlier_fraction": 0.03},
    },
}
