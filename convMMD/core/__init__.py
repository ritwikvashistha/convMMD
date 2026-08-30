from .data import (
    generate_1d_laplace_mixture,
    generate_moons,
    generate_circles,
    generate_checkerboard,
    DATASET_REGISTRY,
)
from .losses import mmd_laplace_kernel, mmd_gaussian_kernel, compute_bandwidth_median_heuristic
from .evaluate import (
    compute_ise,
    sliced_wasserstein_distance,
    compute_swd_scaled,
    compute_mse,
)

__all__ = [
    "DATASET_REGISTRY",
    "compute_bandwidth_median_heuristic",
    "compute_ise",
    "compute_mse",
    "compute_swd_scaled",
    "generate_1d_laplace_mixture",
    "generate_checkerboard",
    "generate_circles",
    "generate_moons",
    "mmd_gaussian_kernel",
    "mmd_laplace_kernel",
    "sliced_wasserstein_distance",
]
