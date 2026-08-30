"""Compatibility and isolation tests for the package-local bounded spline."""

import copy
import importlib
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import nflows.transforms.autoregressive as nflows_autoregressive
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform as NFlowsRQSTransform,
)
from nflows.transforms.splines import rational_quadratic as nflows_rational_quadratic

from convMMD.density_models.nf import (
    BoundedPiecewiseRationalQuadraticAutoregressiveTransform,
    NormalizingFlowDensity,
    _bounded_rational_quadratic_spline,
    _bounded_unconstrained_rational_quadratic_spline,
    create_nsf_1d,
    create_nsf_nd,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _transform_kwargs(features):
    return {
        "features": features,
        "hidden_features": 8,
        "num_bins": 4,
        "tails": "linear",
        "tail_bound": 3.0,
        "num_blocks": 1,
        "use_residual_blocks": False,
        "min_bin_width": 1e-2,
        "min_bin_height": 1e-2,
        "min_derivative": 1e-2,
    }


def _assert_fresh_density_models_import_preserves_nflows():
    for module_name in tuple(sys.modules):
        if module_name == "convMMD.density_models" or module_name.startswith(
            "convMMD.density_models."
        ):
            del sys.modules[module_name]
    sys.modules["convMMD"].__dict__.pop("density_models", None)

    rational_quadratic = importlib.reload(nflows_rational_quadratic)
    autoregressive = importlib.reload(nflows_autoregressive)
    before = (
        autoregressive.rational_quadratic_spline,
        autoregressive.unconstrained_rational_quadratic_spline,
    )

    importlib.import_module("convMMD.density_models")
    after = (
        autoregressive.rational_quadratic_spline,
        autoregressive.unconstrained_rational_quadratic_spline,
    )

    assert before == after
    assert after[0] is rational_quadratic.rational_quadratic_spline
    assert (
        after[1]
        is rational_quadratic.unconstrained_rational_quadratic_spline
    )


def test_fresh_import_does_not_mutate_nflows_spline_functions():
    script = """
import nflows.transforms.autoregressive as autoregressive
from nflows.transforms.splines import rational_quadratic

before = (
    autoregressive.rational_quadratic_spline,
    autoregressive.unconstrained_rational_quadratic_spline,
)
import convMMD.density_models
after = (
    autoregressive.rational_quadratic_spline,
    autoregressive.unconstrained_rational_quadratic_spline,
)

assert before == after
assert after[0] is rational_quadratic.rational_quadratic_spline
assert after[1] is rational_quadratic.unconstrained_rational_quadratic_spline
"""
    child_env = os.environ.copy()
    child_env.update(
        {
            "KMP_USE_SHM": "0",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    try:
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            env=child_env,
            text=True,
            timeout=30,
        )
        return
    except subprocess.CalledProcessError as error:
        if (
            "OMP: Error #179" not in error.stderr
            or "fork" not in multiprocessing.get_all_start_methods()
        ):
            raise

    # Some older local torch/OpenMP builds cannot initialize shared memory in a
    # subprocess after pytest imports torch. A forked child still isolates the
    # module re-import and preserves the same callable-identity assertion.
    context = multiprocessing.get_context("fork")
    process = context.Process(target=_assert_fresh_density_models_import_preserves_nflows)
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("fresh density-model import check timed out")
    assert process.exitcode == 0


@pytest.mark.parametrize("features", [1, 2])
def test_local_transform_matches_v010_legacy_path(features):
    torch.manual_seed(8128 + features)
    legacy = NFlowsRQSTransform(**_transform_kwargs(features))
    local = BoundedPiecewiseRationalQuadraticAutoregressiveTransform(
        **_transform_kwargs(features)
    )
    local.load_state_dict(legacy.state_dict(), strict=True)

    assert list(local.state_dict()) == list(legacy.state_dict())

    inputs = torch.linspace(-0.9, 0.8, steps=3 * features).reshape(3, features)
    legacy_inputs = inputs.clone().requires_grad_(True)

    original_rational = nflows_autoregressive.rational_quadratic_spline
    original_unconstrained = (
        nflows_autoregressive.unconstrained_rational_quadratic_spline
    )
    try:
        nflows_autoregressive.rational_quadratic_spline = (
            _bounded_rational_quadratic_spline
        )
        nflows_autoregressive.unconstrained_rational_quadratic_spline = (
            _bounded_unconstrained_rational_quadratic_spline
        )
        legacy_outputs, legacy_logabsdet = legacy(legacy_inputs)
        legacy_inverse, legacy_inverse_logabsdet = legacy.inverse(
            legacy_outputs.detach()
        )
        legacy_gradient = torch.autograd.grad(
            legacy_outputs.square().sum() + legacy_logabsdet.sum(),
            legacy_inputs,
        )[0]
    finally:
        nflows_autoregressive.rational_quadratic_spline = original_rational
        nflows_autoregressive.unconstrained_rational_quadratic_spline = (
            original_unconstrained
        )

    local_inputs = inputs.clone().requires_grad_(True)
    local_outputs, local_logabsdet = local(local_inputs)
    local_inverse, local_inverse_logabsdet = local.inverse(local_outputs.detach())
    local_gradient = torch.autograd.grad(
        local_outputs.square().sum() + local_logabsdet.sum(),
        local_inputs,
    )[0]

    torch.testing.assert_close(local_outputs, legacy_outputs, rtol=0, atol=0)
    torch.testing.assert_close(local_logabsdet, legacy_logabsdet, rtol=0, atol=0)
    torch.testing.assert_close(local_inverse, legacy_inverse, rtol=0, atol=0)
    torch.testing.assert_close(
        local_inverse_logabsdet,
        legacy_inverse_logabsdet,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(local_gradient, legacy_gradient, rtol=0, atol=0)

    assert nflows_autoregressive.rational_quadratic_spline is original_rational
    assert (
        nflows_autoregressive.unconstrained_rational_quadratic_spline
        is original_unconstrained
    )


def test_derivative_cap_is_instance_local_and_deepcopy_safe():
    cap_three = BoundedPiecewiseRationalQuadraticAutoregressiveTransform(
        **_transform_kwargs(1), max_derivative=3.0
    )
    cap_ten = BoundedPiecewiseRationalQuadraticAutoregressiveTransform(
        **_transform_kwargs(1), max_derivative=10.0
    )
    cap_ten.load_state_dict(cap_three.state_dict(), strict=True)

    with torch.no_grad():
        for parameter in cap_three.parameters():
            parameter.zero_()
        cap_ten.load_state_dict(cap_three.state_dict(), strict=True)

    inputs = torch.tensor([[-0.75], [-0.25], [0.25], [0.75]])
    outputs_three, _ = cap_three(inputs)
    outputs_ten, _ = cap_ten(inputs)
    copied = copy.deepcopy(cap_three)

    assert cap_three.max_derivative == 3.0
    assert cap_ten.max_derivative == 10.0
    assert copied.max_derivative == 3.0
    assert not torch.allclose(outputs_three, outputs_ten)
    torch.testing.assert_close(copied(inputs)[0], outputs_three)


@pytest.mark.parametrize("max_derivative", [1.0, 0.0, float("nan"), float("inf")])
def test_local_transform_rejects_invalid_derivative_cap(max_derivative):
    with pytest.raises(ValueError, match="greater than 1"):
        BoundedPiecewiseRationalQuadraticAutoregressiveTransform(
            **_transform_kwargs(1), max_derivative=max_derivative
        )


def test_package_factories_use_local_transform_and_preserve_state_keys():
    flow_1d = create_nsf_1d(
        num_blocks=1,
        num_bins=4,
        hidden_features=8,
        tail_bound=4.0,
    )
    flow_nd = create_nsf_nd(
        dim=2,
        num_blocks=1,
        num_bins=4,
        hidden_features=8,
        tail_bound=3.0,
    )
    wrapper = NormalizingFlowDensity(
        dim=2,
        num_blocks=1,
        num_bins=4,
        hidden_features=8,
        max_derivative=3.0,
    )

    assert isinstance(
        flow_1d._transform._transforms[0],
        BoundedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
    assert isinstance(
        flow_nd._transform._transforms[0],
        BoundedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
    assert isinstance(
        wrapper.flow._transform._transforms[0],
        BoundedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
    assert wrapper.flow._transform._transforms[0].max_derivative == 3.0

    torch.manual_seed(144)
    legacy = NFlowsRQSTransform(**_transform_kwargs(2))
    local = BoundedPiecewiseRationalQuadraticAutoregressiveTransform(
        **_transform_kwargs(2)
    )
    load_result = local.load_state_dict(legacy.state_dict(), strict=True)

    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    assert list(local.state_dict()) == list(legacy.state_dict())


def test_upstream_transform_remains_unmodified():
    assert (
        nflows_autoregressive.rational_quadratic_spline
        is nflows_rational_quadratic.rational_quadratic_spline
    )
    assert (
        nflows_autoregressive.unconstrained_rational_quadratic_spline
        is nflows_rational_quadratic.unconstrained_rational_quadratic_spline
    )
