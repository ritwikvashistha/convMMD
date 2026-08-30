"""Round-trip and validation tests for versioned flow checkpoints."""

import io

import numpy as np
import pytest
import torch

import convMMD
from convMMD.density_models import (
    load_normalizing_flow_checkpoint,
    save_normalizing_flow_checkpoint,
)
from convMMD.density_models.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
)
from convMMD.density_models.nf import NormalizingFlowDensity


def _nsf_model(dim):
    kwargs = {
        "dim": dim,
        "flow_type": "nsf",
        "num_blocks": 1,
        "num_bins": 4,
        "hidden_features": 8,
        "tail_bound": 4.0 if dim == 1 else 3.0,
        "max_derivative": 3.0,
    }
    if dim == 2:
        kwargs.update(
            data_mean=np.array([0.25, -0.5]),
            data_std=np.array([1.5, 0.75]),
        )
    return NormalizingFlowDensity(**kwargs)


def _iaf_model():
    return NormalizingFlowDensity(
        dim=2,
        flow_type="iaf",
        num_blocks=2,
        num_bins=7,
        hidden_features=8,
        tail_bound=5.0,
        data_mean=[0.25, -0.5],
        data_std=[1.5, 0.75],
        max_derivative=6.0,
        num_blocks_per_layer=1,
        dropout_prob=0.0,
        use_residual_blocks=False,
        use_random_permutations=False,
        use_random_masks=True,
        batch_norm_within_layers=False,
        batch_norm_between_layers=True,
        use_actnorm=False,
    )


def _stateful_iaf_model():
    return NormalizingFlowDensity(
        dim=2,
        flow_type="iaf",
        num_blocks=2,
        hidden_features=8,
        num_blocks_per_layer=1,
        dropout_prob=0.0,
        use_residual_blocks=False,
        use_random_permutations=True,
        use_random_masks=False,
        batch_norm_within_layers=True,
        batch_norm_between_layers=False,
        use_actnorm=True,
    )


def _assert_plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for item in value:
            _assert_plain(item)
        return
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for item in value.values():
            _assert_plain(item)
        return
    pytest.fail(f"non-plain checkpoint value: {type(value)!r}")


def _assert_same_model(original, loaded, inputs):
    assert original._get_checkpoint_model_config() == (
        loaded._get_checkpoint_model_config()
    )
    assert list(original.state_dict()) == list(loaded.state_dict())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)

    original.eval()
    loaded.eval()
    torch.testing.assert_close(
        original.log_prob(inputs), loaded.log_prob(inputs), rtol=0, atol=0
    )
    torch.manual_seed(919)
    original_samples = original.sample(5)
    torch.manual_seed(919)
    loaded_samples = loaded.sample(5)
    torch.testing.assert_close(original_samples, loaded_samples, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [1, 2])
def test_nsf_checkpoint_round_trip_with_path_and_bytes_io(tmp_path, dim):
    torch.manual_seed(100 + dim)
    model = _nsf_model(dim)
    destination = tmp_path / "flow.pt" if dim == 1 else io.BytesIO()

    save_normalizing_flow_checkpoint(model, destination)
    if hasattr(destination, "seek"):
        destination.seek(0)
    loaded = load_normalizing_flow_checkpoint(destination)

    inputs = torch.linspace(-0.75, 0.75, steps=3 * dim).reshape(3, dim)
    _assert_same_model(model, loaded, inputs)
    assert loaded.max_derivative == 3.0
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())


def test_iaf_checkpoint_restores_every_effective_option(tmp_path):
    torch.manual_seed(211)
    model = _iaf_model()
    checkpoint = tmp_path / "iaf.pt"

    save_normalizing_flow_checkpoint(model, checkpoint)
    loaded = load_normalizing_flow_checkpoint(
        checkpoint, map_location=torch.device("cpu")
    )

    _assert_same_model(model, loaded, torch.tensor([[-0.5, 0.25], [0.5, -0.25]]))
    config = loaded._get_checkpoint_model_config()
    assert set(config["iaf_kwargs"]) == {
        "num_blocks_per_layer",
        "dropout_prob",
        "use_residual_blocks",
        "use_random_permutations",
        "use_random_masks",
        "batch_norm_within_layers",
        "batch_norm_between_layers",
        "use_actnorm",
    }
    assert config["iaf_kwargs"]["use_random_masks"] is True
    assert config["iaf_kwargs"]["batch_norm_between_layers"] is True


def test_iaf_checkpoint_restores_actnorm_permutations_and_inner_batchnorm(tmp_path):
    torch.manual_seed(317)
    model = _stateful_iaf_model()
    initialization_inputs = torch.tensor(
        [[-1.0, 0.5], [-0.25, -0.5], [0.5, 1.0], [1.25, -1.0]]
    )
    model.log_prob(initialization_inputs)
    checkpoint = tmp_path / "stateful-iaf.pt"

    save_normalizing_flow_checkpoint(model, checkpoint)
    loaded = load_normalizing_flow_checkpoint(checkpoint)

    _assert_same_model(model, loaded, initialization_inputs)
    config = loaded._get_checkpoint_model_config()["iaf_kwargs"]
    assert config["use_actnorm"] is True
    assert config["use_random_permutations"] is True
    assert config["batch_norm_within_layers"] is True


def test_checkpoint_payload_is_weights_only_readable_and_plain(tmp_path):
    model = _nsf_model(2)
    checkpoint = tmp_path / "flow.pt"
    save_normalizing_flow_checkpoint(model, checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["package_version"] == convMMD.__version__
    assert payload["model_type"] == "NormalizingFlowDensity"
    assert payload["model_dtype"] == "float32"
    assert payload["model_config"]["data_mean"] == [0.25, -0.5]
    assert payload["model_config"]["data_std"] == [1.5, 0.75]
    assert payload["model_config"]["max_derivative"] == 3.0
    _assert_plain(payload["model_config"])


def test_float64_checkpoint_round_trip_preserves_dtype(tmp_path):
    torch.manual_seed(341)
    model = _nsf_model(1).double()
    checkpoint = tmp_path / "float64.pt"

    save_normalizing_flow_checkpoint(model, checkpoint)
    loaded = load_normalizing_flow_checkpoint(checkpoint)

    assert {
        value.dtype
        for value in loaded.state_dict().values()
        if value.is_floating_point()
    } == {torch.float64}
    inputs = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float64)
    _assert_same_model(model, loaded, inputs)


def test_save_rejects_mixed_floating_dtypes(tmp_path):
    model = _nsf_model(1)
    first_parameter = next(model.parameters())
    first_parameter.data = first_parameter.data.double()

    with pytest.raises(ValueError, match="exactly one floating-point dtype"):
        save_normalizing_flow_checkpoint(model, tmp_path / "mixed.pt")


def test_save_rejects_mutated_derivative_policy(tmp_path):
    model = _nsf_model(1)
    model.flow._transform._transforms[0].max_derivative = 10.0

    with pytest.raises(ValueError, match="max_derivative has drifted"):
        save_normalizing_flow_checkpoint(model, tmp_path / "drifted.pt")


def _save_tampered_payload(tmp_path, mutate):
    source = tmp_path / "source.pt"
    target = tmp_path / "tampered.pt"
    save_normalizing_flow_checkpoint(_nsf_model(1), source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, target)
    return target


def _replace_first_floating_tensor_with_integer(payload):
    for key, value in payload["state_dict"].items():
        if value.is_floating_point():
            payload["state_dict"][key] = value.to(torch.int64)
            return
    raise AssertionError("test checkpoint has no floating tensors")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.__setitem__("format_version", 99),
            "Unsupported checkpoint format version",
        ),
        (
            lambda payload: payload["model_config"].pop("max_derivative"),
            "missing keys.*max_derivative",
        ),
        (
            lambda payload: payload["model_config"].__setitem__("typo", 3),
            "unexpected keys.*typo",
        ),
        (
            lambda payload: payload.__setitem__("model_dtype", "float64"),
            "does not match",
        ),
        (
            lambda payload: payload.__setitem__("model_dtype", []),
            "model_dtype must be a string",
        ),
        (
            lambda payload: payload["model_config"].__setitem__("flow_type", []),
            "flow_type must be a string",
        ),
        (
            lambda payload: payload["model_config"].__setitem__("tail_bound", -1),
            "tail_bound must be strictly positive",
        ),
        (
            lambda payload: payload["model_config"].__setitem__(
                "data_mean", [0.0, 1.0]
            ),
            "data_mean must be scalar or have length dim",
        ),
        (
            _replace_first_floating_tensor_with_integer,
            "tensor dtype mismatch",
        ),
        (
            lambda payload: payload["model_config"].__setitem__("num_bins", 5),
            "tensor shape mismatch",
        ),
    ],
)
def test_load_rejects_tampered_checkpoint(tmp_path, mutate, message):
    checkpoint = _save_tampered_payload(tmp_path, mutate)

    with pytest.raises(ValueError, match=message):
        load_normalizing_flow_checkpoint(checkpoint)


def test_load_rejects_raw_state_dict_with_migration_guidance(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    torch.save(_nsf_model(1).state_dict(), checkpoint)

    with pytest.raises(ValueError, match="v0.1.0.*explicit configuration"):
        load_normalizing_flow_checkpoint(checkpoint)


def test_load_rejects_non_mapping_payload(tmp_path):
    checkpoint = tmp_path / "list.pt"
    torch.save([1, 2, 3], checkpoint)

    with pytest.raises(ValueError, match="payload must be a mapping"):
        load_normalizing_flow_checkpoint(checkpoint)


def test_save_rejects_other_model_types(tmp_path):
    with pytest.raises(TypeError, match="NormalizingFlowDensity"):
        save_normalizing_flow_checkpoint(torch.nn.Linear(1, 1), tmp_path / "bad.pt")
