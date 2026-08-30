"""
2D Deconvolution Experiment

Validates convMMD on 2D deconvolution tasks (moons, circles, checkerboard) with
heteroscedastic Gaussian noise and optional observation contamination. The
default `outlier_fraction` is 0.03; use 0 for the pure additive-error model.

Usage:
    python examples/deconv_2d.py --dataset moons --flow_type nsf --epochs 500
    python examples/deconv_2d.py --dataset checkerboard --flow_type iaf --epochs 500
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from convMMD.core.data import (
    generate_checkerboard,
    generate_circles,
    generate_moons,
)
from convMMD.core.evaluate import sliced_wasserstein_distance
from convMMD.density_models import (
    NormalizingFlowDensity,
    save_normalizing_flow_checkpoint,
)
from convMMD.training import train_convmmd


def get_dataset(
    name: str,
    n_samples: int,
    seed: int,
    device: str,
    outlier_fraction: float = 0.03,
):
    """Get dataset by name."""
    if name == "moons":
        generator = generate_moons
    elif name == "circles":
        generator = generate_circles
    elif name == "checkerboard":
        generator = generate_checkerboard
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return generator(
        n_samples=n_samples,
        seed=seed,
        device=device,
        outlier_fraction=outlier_fraction,
    )


def evaluate_model(
    model,
    theta_true,
    device,
    n_eval_samples=5000,
    n_projections=1000,
    seed=0,
):
    """Evaluate model using reproducible simulation-only SWD."""
    model.eval()
    resolved_device = torch.device(device)
    cuda_index = None
    if resolved_device.type == "cuda":
        cuda_index = (
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        )
    cuda_devices = [] if cuda_index is None else [cuda_index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if cuda_index is not None:
            with torch.cuda.device(cuda_index):
                torch.cuda.manual_seed(seed)
        with torch.no_grad():
            samples = model.sample(n_eval_samples).to(device)

    swd = sliced_wasserstein_distance(
        samples,
        theta_true,
        n_projections=n_projections,
        seed=seed + 1,
    )
    swd_scaled = swd * np.sqrt(samples.shape[1])

    return {"swd": swd, "swd_scaled": swd_scaled, "samples": samples}


def plot_results(
    theta_true,
    x_noisy,
    result,
    history,
    dataset_name,
    save_path=None,
):
    """Plot density estimates and training history."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    samples = result["samples"].cpu().numpy()
    theta_np = theta_true.cpu().numpy()
    x_noisy_np = x_noisy.cpu().numpy()

    # Row 1: Data comparison
    # Clean data
    ax = axes[0, 0]
    ax.scatter(theta_np[:, 0], theta_np[:, 1], s=3, alpha=0.5, c="green")
    ax.set_title(f"{dataset_name.title()} - True Latent")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.axis("equal")

    # Noisy observations
    ax = axes[0, 1]
    ax.scatter(x_noisy_np[:, 0], x_noisy_np[:, 1], s=3, alpha=0.5, c="gray")
    ax.set_title("Noisy Observations")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.axis("equal")

    # Generated samples
    ax = axes[1, 0]
    ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5, c="blue")
    ax.set_title(f"Flow Samples (SWD: {result['swd']:.4f})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.axis("equal")

    # Training loss
    ax = axes[1, 1]
    if history["loss"]:
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

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="2D Deconvolution Experiment")
    parser.add_argument("--dataset", type=str, default="moons",
                        choices=["moons", "circles", "checkerboard"])
    parser.add_argument("--flow_type", type=str, default="nsf",
                        choices=["nsf", "iaf"])
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--num_bins", type=int, default=8)
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--n_eval_samples", type=int, default=5000)
    parser.add_argument("--n_projections", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outlier_fraction", type=float, default=0.03)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_plot", type=str, default=None)
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default=None,
        help="Optional path for a versioned NormalizingFlowDensity checkpoint",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset}, Flow: {args.flow_type}")
    print(f"Settings: N={args.n_samples}, epochs={args.epochs}")

    # Generate data
    theta_true, x_noisy, noise_std = get_dataset(
        args.dataset, args.n_samples, args.seed, args.device, args.outlier_fraction
    )

    print(f"Data: theta range x1=[{theta_true[:, 0].min():.2f}, {theta_true[:, 0].max():.2f}], "
          f"x2=[{theta_true[:, 1].min():.2f}, {theta_true[:, 1].max():.2f}]")

    # Fit uses only noisy observations and known error standard deviations.
    torch.manual_seed(args.seed + 1)
    data_mean = x_noisy.mean(dim=0).detach().cpu().numpy()
    data_std = x_noisy.std(dim=0).detach().cpu().numpy()

    # Create model
    model = NormalizingFlowDensity(
        dim=2,
        flow_type=args.flow_type,
        num_blocks=args.num_blocks,
        num_bins=args.num_bins,
        hidden_features=args.hidden_features,
        tail_bound=3.0,
        data_mean=data_mean,
        data_std=data_std,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    result = train_convmmd(
        model=model,
        x_noisy=x_noisy,
        noise_std=noise_std,
        noise_type="gaussian",
        kernel_type="laplace",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=0,
        max_grad_norm=1.0,
        eval_every=min(50, args.epochs),
        device=args.device,
        verbose=True,
    )

    # Synthetic truth enters only after fitting, for evaluation and plotting.
    final_eval = evaluate_model(
        result["model"],
        theta_true,
        args.device,
        n_eval_samples=args.n_eval_samples,
        n_projections=args.n_projections,
        seed=args.seed + 2,
    )
    print(f"\nFinal SWD: {final_eval['swd']:.6f}")
    print(f"Final SWD (scaled by sqrt(d)): {final_eval['swd_scaled']:.6f}")

    if args.save_checkpoint:
        save_normalizing_flow_checkpoint(result["model"], args.save_checkpoint)
        print(f"Saved checkpoint to {args.save_checkpoint}")

    # Plot results
    plot_results(
        theta_true,
        x_noisy,
        final_eval,
        result["history"],
        args.dataset,
        save_path=args.save_plot,
    )


if __name__ == "__main__":
    main()
