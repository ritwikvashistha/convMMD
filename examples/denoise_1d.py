"""Minimal one-dimensional posterior-mean denoising example.

This synthetic example supplies the Gaussian measurement-error standard
deviation to ``convMMD.denoise``. The clean values are used only to report a
diagnostic MSE; they are not used for fitting or denoising.

Usage:
    python examples/denoise_1d.py --device cpu --n_samples 64 --epochs 2
"""

import argparse

import torch

from convMMD import denoise
from convMMD.core.data import generate_1d_laplace_mixture


def main():
    parser = argparse.ArgumentParser(
        description="Low-dimensional convMMD posterior-mean denoising"
    )
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.5,
        help="Known Gaussian measurement-error standard deviation",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_bins", type=int, default=16)
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument(
        "--posterior_method",
        choices=("importance", "langevin"),
        default="importance",
    )
    parser.add_argument("--num_importance_samples", type=int, default=8192)
    parser.add_argument("--posterior_batch_size", type=int, default=64)
    parser.add_argument("--langevin_steps", type=int, default=1000)
    parser.add_argument("--langevin_step_size", type=float, default=1e-2)
    parser.add_argument("--langevin_chains", type=int, default=100)
    parser.add_argument("--langevin_burn_in_fraction", type=float, default=0.6)
    parser.add_argument("--langevin_thinning", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default=None,
        help="Optional path for the fitted flow only; no data or history is saved",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the training progress display",
    )
    args = parser.parse_args()

    theta_true, observations, known_noise_std = generate_1d_laplace_mixture(
        n_samples=args.n_samples,
        noise_type="gaussian",
        noise_std_range=(args.noise_std, args.noise_std),
        seed=args.seed,
        device=args.device,
    )

    result = denoise(
        observations,
        known_noise_std,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_blocks=args.num_blocks,
        num_bins=args.num_bins,
        hidden_features=args.hidden_features,
        posterior_method=args.posterior_method,
        num_importance_samples=args.num_importance_samples,
        posterior_batch_size=args.posterior_batch_size,
        langevin_steps=args.langevin_steps,
        langevin_step_size=args.langevin_step_size,
        langevin_chains=args.langevin_chains,
        langevin_burn_in_fraction=args.langevin_burn_in_fraction,
        langevin_thinning=args.langevin_thinning,
        seed=args.seed,
        device=args.device,
        verbose=not args.quiet,
    )

    noisy_mse = float((observations - theta_true).square().mean())
    denoised_mse = float((result.denoised - theta_true).square().mean())
    print(f"Denoised shape: {tuple(result.denoised.shape)}")
    print(f"Noisy coordinatewise MSE: {noisy_mse:.6f}")
    print(f"Denoised coordinatewise MSE: {denoised_mse:.6f}")
    print("These synthetic diagnostics are not a quality benchmark.")

    if args.save_checkpoint:
        result.save(args.save_checkpoint)
        print(f"Saved fitted-flow-only checkpoint to {args.save_checkpoint}")


if __name__ == "__main__":
    main()
