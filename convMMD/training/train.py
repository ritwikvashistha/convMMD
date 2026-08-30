"""
Training loop for convMMD estimation.
"""

import torch
import torch.optim as optim
import numpy as np
import math
from typing import Dict, Any, Callable, Optional
from tqdm import tqdm

from ..core.losses import (
    BandwidthInput,
    _normalize_bandwidths,
    compute_bandwidth_median_heuristic,
    mmd_gaussian_kernel,
    mmd_laplace_kernel,
)


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


def _validate_training_data(x_noisy, noise_std):
    if not torch.is_tensor(x_noisy):
        raise TypeError("x_noisy must be a PyTorch tensor")
    if x_noisy.ndim != 2:
        raise ValueError("x_noisy must have shape (n, d)")
    if x_noisy.shape[0] < 2:
        raise ValueError("train_convmmd requires at least two observations")
    if x_noisy.shape[1] < 1:
        raise ValueError("x_noisy must contain at least one feature")
    if not x_noisy.is_floating_point():
        raise TypeError("x_noisy must use a floating-point dtype")
    if not torch.isfinite(x_noisy).all().item():
        raise ValueError("x_noisy must contain only finite values")

    if not torch.is_tensor(noise_std):
        raise TypeError("noise_std must be a PyTorch tensor")
    if noise_std.ndim != 2 or noise_std.shape[0] != x_noisy.shape[0]:
        raise ValueError("noise_std must have shape (n, 1) or (n, d)")
    if noise_std.shape[1] not in (1, x_noisy.shape[1]):
        raise ValueError("noise_std must have shape (n, 1) or (n, d)")
    if not noise_std.is_floating_point():
        raise TypeError("noise_std must use a floating-point dtype")
    if noise_std.dtype != x_noisy.dtype:
        raise TypeError("noise_std and x_noisy must use the same dtype")
    if not torch.isfinite(noise_std).all().item():
        raise ValueError("noise_std must contain only finite values")
    if (noise_std < 0).any().item():
        raise ValueError("noise_std must be nonnegative")
    return x_noisy.detach(), noise_std.detach()


def _validate_model_samples(samples, *, sample_count, reference):
    expected_shape = (sample_count, reference.shape[1])
    if not torch.is_tensor(samples):
        raise TypeError("model.sample must return a PyTorch tensor")
    if samples.shape != expected_shape:
        raise ValueError(f"model.sample must return shape {expected_shape}")
    if samples.device != reference.device:
        raise ValueError("model.sample must return samples on the training device")
    if samples.dtype != reference.dtype:
        raise TypeError("model samples and x_noisy must use the same dtype")
    return samples


def train_convmmd(
    model: torch.nn.Module,
    x_noisy: torch.Tensor,
    noise_std: torch.Tensor,
    noise_type: str = "laplace",
    kernel_type: str = "laplace",
    epochs: int = 1000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = None,
    max_grad_norm: float = 5.0,
    bandwidths: Optional[BandwidthInput] = None,
    eval_fn: Optional[Callable] = None,
    eval_every: int = 50,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a density model using convMMD objective.

    Args:
        model: Density model (e.g., NormalizingFlowDensity)
        x_noisy: Noisy observations (n, d)
        noise_std: Per-sample noise standard deviations with shape (n, 1) for
            isotropic errors or (n, d) for known diagonal errors
        noise_type: Type of noise to simulate ("laplace" or "gaussian")
        kernel_type: Type of kernel for MMD ("laplace" or "gaussian")
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        warmup_epochs: Number of warmup epochs (default: 10% of epochs)
        max_grad_norm: Positive gradient-clipping norm.
        bandwidths: Kernel bandwidths (computed automatically if None)
        eval_fn: Optional diagnostic function called every ``eval_every``
            epochs. Its return value is recorded but never used for fitting.
            PyTorch RNG state is restored after the callback; the callback must
            not mutate the model.
        eval_every: Evaluate and record history every N epochs
        device: Device to train on
        verbose: Print progress

    Returns:
        Dictionary with training history, effective bandwidths, and the fitted
        model in evaluation mode.
    """
    if noise_type not in ("laplace", "gaussian"):
        raise ValueError(f"Unknown noise type: {noise_type}")
    if kernel_type not in ("laplace", "gaussian"):
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    x_noisy, noise_std = _validate_training_data(x_noisy, noise_std)
    epochs = _validate_integer(epochs, name="epochs", minimum=1)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 2
    ):
        raise ValueError(
            "batch_size must be at least 2 for the unbiased MMD U-statistic"
        )
    eval_every = _validate_integer(eval_every, name="eval_every", minimum=1)
    lr = _validate_real(lr, name="lr", lower_bound=0.0, strict=True)
    weight_decay = _validate_real(
        weight_decay, name="weight_decay", lower_bound=0.0, strict=False
    )
    max_grad_norm = _validate_real(
        max_grad_norm, name="max_grad_norm", lower_bound=0.0, strict=True
    )
    if warmup_epochs is not None:
        warmup_epochs = _validate_integer(
            warmup_epochs, name="warmup_epochs", minimum=0
        )
        if warmup_epochs > epochs:
            raise ValueError("warmup_epochs must not exceed epochs")
    if not isinstance(model, torch.nn.Module) or not callable(
        getattr(model, "sample", None)
    ):
        raise TypeError("model must be a PyTorch module with a callable sample method")

    model = model.to(device)
    x_noisy = x_noisy.to(device)
    noise_std = noise_std.to(device)

    n_samples = x_noisy.shape[0]

    # Handle batch size >= n_samples (full batch training)
    if batch_size >= n_samples:
        batch_size = n_samples
        n_batches = 1
    else:
        n_batches = n_samples // batch_size

    # Select MMD function based on kernel type
    if kernel_type == "laplace":
        mmd_fn = mmd_laplace_kernel
    else:
        mmd_fn = mmd_gaussian_kernel

    # Compute bandwidths if not provided
    if bandwidths is None:
        with torch.no_grad():
            n_bw = min(2000, n_samples)
            samples_init = _validate_model_samples(
                model.sample(n_bw),
                sample_count=n_bw,
                reference=x_noisy,
            )
            if noise_type == "laplace":
                noise_scale = noise_std[:n_bw] / np.sqrt(2.0)
                noise = (torch.rand_like(samples_init) - 0.5).sign() * torch.log(1 - 2 * torch.abs(torch.rand_like(samples_init) - 0.5) + 1e-10) * noise_scale
            else:
                noise = torch.randn_like(samples_init) * noise_std[:n_bw]
            noisy_samples = samples_init + noise
            bandwidths = compute_bandwidth_median_heuristic(x_noisy[:n_bw], noisy_samples)
        if verbose:
            print(f"Computed bandwidths: {bandwidths.cpu().numpy()}")

    bandwidths = _normalize_bandwidths(bandwidths, x_noisy)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup + cosine decay schedule
    if warmup_epochs is None:
        warmup_epochs = min(200, epochs // 10)

    total_steps = epochs
    warmup_steps = warmup_epochs

    history = {"epoch": [], "loss": [], "eval": []}

    if verbose:
        print(f"Training for {epochs} epochs with batch_size={batch_size}, lr={lr}")
        print(f"Warmup: {warmup_steps} epochs, then cosine decay")

    pbar = tqdm(range(epochs), disable=not verbose)
    for epoch in pbar:
        model.train()

        # Learning rate schedule: linear warmup then cosine decay
        if epoch < warmup_steps:
            lr_now = lr * (epoch + 1) / warmup_steps
        else:
            frac = (epoch - warmup_steps) / max(1, total_steps - warmup_steps)
            lr_now = lr * 0.5 * (1.0 + math.cos(math.pi * frac))

        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # Shuffle data
        perm = torch.randperm(n_samples, device=device)
        x_shuffled = x_noisy[perm]
        noise_std_shuffled = noise_std[perm]

        epoch_loss = 0.0
        finite_steps = 0
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_batch = x_shuffled[start:end]
            noise_std_batch = noise_std_shuffled[start:end]

            optimizer.zero_grad()

            # Sample from model
            model_samples = _validate_model_samples(
                model.sample(batch_size),
                sample_count=batch_size,
                reference=x_noisy,
            )

            # Add simulated noise to match the observation process
            if noise_type == "laplace":
                noise_scale = noise_std_batch / np.sqrt(2.0)
                sim_noise = (torch.rand_like(model_samples) - 0.5).sign() * torch.log(1 - 2 * torch.abs(torch.rand_like(model_samples) - 0.5) + 1e-10) * noise_scale
            else:
                sim_noise = torch.randn_like(model_samples) * noise_std_batch

            noisy_model_samples = model_samples + sim_noise

            # Compute MMD loss
            loss = mmd_fn(noisy_model_samples, x_batch, bandwidths)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            finite_steps += 1

        if finite_steps == 0:
            raise RuntimeError(
                f"No finite optimization steps completed in epoch {epoch + 1}"
            )
        avg_loss = epoch_loss / finite_steps

        # Evaluation
        if (epoch + 1) % eval_every == 0:
            history["epoch"].append(epoch + 1)
            history["loss"].append(avg_loss)

            if eval_fn is not None:
                model.eval()
                cuda_index = None
                if x_noisy.device.type == "cuda":
                    cuda_index = (
                        x_noisy.device.index
                        if x_noisy.device.index is not None
                        else torch.cuda.current_device()
                    )
                cuda_devices = [] if cuda_index is None else [cuda_index]
                with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                    eval_result = eval_fn(model)
                history["eval"].append(eval_result)
                # Handle both scalar and dict eval results
                if isinstance(eval_result, dict):
                    eval_display = eval_result.get("swd", eval_result.get("ise", 0))
                else:
                    eval_display = eval_result
                pbar.set_postfix({"loss": f"{avg_loss:.6f}", "eval": f"{eval_display:.4f}", "lr": f"{lr_now:.2e}"})
            else:
                history["eval"].append(None)
                pbar.set_postfix({"loss": f"{avg_loss:.6f}", "lr": f"{lr_now:.2e}"})

    model.eval()
    return {
        "model": model,
        "history": history,
        "bandwidths": bandwidths.detach().cpu(),
    }
