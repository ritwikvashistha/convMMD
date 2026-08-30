"""
1D Deconvolution Experiment

Validates convMMD on a 1D deconvolution task with a Laplace mixture latent
density and heteroscedastic noise, using a single sample size and single run.

Usage:
    python examples/deconv_1d.py --n_samples 8000 --epochs 500
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from convMMD.core.data import (
    generate_1d_laplace_mixture,
    true_density_1d_laplace_mixture,
)
from convMMD.core.evaluate import compute_ise
from convMMD.density_models import (
    NormalizingFlowDensity,
    save_normalizing_flow_checkpoint,
)
from convMMD.training import train_convmmd


def plot_results(model, x_noisy, theta_true, history, save_path=None, device="cpu"):
    """Plot density estimates and training history."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 1. Density comparison
    ax = axes[0]
    x_grid = np.linspace(-8, 10, 500)
    true_pdf = true_density_1d_laplace_mixture(x_grid)

    x_tensor = torch.tensor(x_grid, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        log_probs = model.log_prob(x_tensor)
        est_pdf = torch.exp(log_probs).cpu().numpy()

    ax.hist(x_noisy.cpu().numpy().flatten(), bins=60, density=True, alpha=0.3, color="gray", label="Noisy Obs")
    ax.hist(theta_true.cpu().numpy().flatten(), bins=60, density=True, alpha=0.3, color="green", label="True Samples")
    ax.plot(x_grid, true_pdf, "g--", linewidth=2, label="True PDF")
    ax.plot(x_grid, est_pdf, "b-", linewidth=2, label="Flow PDF")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Density Estimation")
    ax.legend()

    # 2. Training loss
    ax = axes[1]
    ax.plot(history["epoch"], history["loss"], "b-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("symlog", linthresh=1e-4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="1D Deconvolution Experiment")
    parser.add_argument("--n_samples", type=int, default=8000, help="Number of samples")
    parser.add_argument("--noise_type", type=str, default="laplace", choices=["laplace", "gaussian"])
    parser.add_argument("--kernel_type", type=str, default="laplace", choices=["laplace", "gaussian"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_bins", type=int, default=16)
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_plot", type=str, default=None)
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default=None,
        help="Optional path for a versioned NormalizingFlowDensity checkpoint",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Settings: N={args.n_samples}, noise={args.noise_type}, kernel={args.kernel_type}")

    # Generate data
    theta_true, x_noisy, noise_std = generate_1d_laplace_mixture(
        n_samples=args.n_samples,
        noise_type=args.noise_type,
        seed=args.seed,
        device=args.device,
    )

    print(f"Data: theta range [{theta_true.min():.2f}, {theta_true.max():.2f}]")
    print(f"Data: x_noisy range [{x_noisy.min():.2f}, {x_noisy.max():.2f}]")

    # Fit uses only noisy observations and known error standard deviations.
    torch.manual_seed(args.seed + 1)
    data_mean = float(x_noisy.mean())
    data_std = float(x_noisy.std())

    # Create model
    model = NormalizingFlowDensity(
        num_blocks=args.num_blocks,
        num_bins=args.num_bins,
        hidden_features=args.hidden_features,
        tail_bound=30.0,
        data_mean=data_mean,
        data_std=data_std,
    )

    result = train_convmmd(
        model=model,
        x_noisy=x_noisy,
        noise_std=noise_std,
        noise_type=args.noise_type,
        kernel_type=args.kernel_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=min(50, args.epochs),
        device=args.device,
        verbose=True,
    )

    # Synthetic truth enters only after fitting, for evaluation and plotting.
    final_ise = compute_ise(
        result["model"],
        true_density_1d_laplace_mixture,
        device=args.device,
    )
    print(f"\nFinal ISE: {final_ise:.6f}")

    if args.save_checkpoint:
        save_normalizing_flow_checkpoint(result["model"], args.save_checkpoint)
        print(f"Saved checkpoint to {args.save_checkpoint}")

    # Plot results
    plot_results(
        result["model"],
        x_noisy,
        theta_true,
        result["history"],
        save_path=args.save_plot,
        device=args.device,
    )


if __name__ == "__main__":
    main()
