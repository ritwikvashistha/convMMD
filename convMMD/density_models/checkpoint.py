"""Versioned, model-only checkpoints for ``NormalizingFlowDensity``."""

from collections.abc import Mapping
import math

import torch

from convMMD import __version__ as _PACKAGE_VERSION

from .nf import (
    BoundedPiecewiseRationalQuadraticAutoregressiveTransform,
    NormalizingFlowDensity,
)


CHECKPOINT_FORMAT = "convMMD.normalizing_flow"
CHECKPOINT_FORMAT_VERSION = 1
_MODEL_TYPE = "NormalizingFlowDensity"

_MODEL_CONFIG_KEYS = {
    "dim",
    "flow_type",
    "num_blocks",
    "num_bins",
    "hidden_features",
    "tail_bound",
    "data_mean",
    "data_std",
    "max_derivative",
    "iaf_kwargs",
}
_IAF_CONFIG_KEYS = {
    "num_blocks_per_layer",
    "dropout_prob",
    "use_residual_blocks",
    "use_random_permutations",
    "use_random_masks",
    "batch_norm_within_layers",
    "batch_norm_between_layers",
    "use_actnorm",
}
_PAYLOAD_KEYS = {
    "format",
    "format_version",
    "package_version",
    "model_type",
    "model_config",
    "model_dtype",
    "state_dict",
}
_DTYPE_TO_NAME = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


def _describe_key_difference(actual, expected):
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing keys: {missing}")
    if unexpected:
        details.append(f"unexpected keys: {unexpected}")
    return "; ".join(details)


def _validate_plain_value(value, path):
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_plain_value(item, f"{path}[{index}]")
        return
    raise ValueError(f"{path} must contain only plain Python values")


def _validate_data_statistic(value, *, name, dim, positive):
    if value is None:
        return
    values = value if isinstance(value, list) else [value]
    if isinstance(value, list) and len(value) != dim:
        raise ValueError(f"model_config.{name} must be scalar or have length dim")
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"model_config.{name} must contain numeric values")
        if not math.isfinite(item):
            raise ValueError(f"model_config.{name} must contain finite values")
        if positive and item <= 0:
            raise ValueError(f"model_config.{name} must be strictly positive")


def _validate_model_config(model_config):
    if not isinstance(model_config, Mapping):
        raise ValueError("model_config must be a mapping")

    if not all(isinstance(key, str) for key in model_config):
        raise ValueError("model_config keys must be strings")
    config_keys = set(model_config)
    if config_keys != _MODEL_CONFIG_KEYS:
        difference = _describe_key_difference(config_keys, _MODEL_CONFIG_KEYS)
        raise ValueError(f"Invalid model_config schema ({difference})")

    flow_type = model_config["flow_type"]
    if not isinstance(flow_type, str):
        raise ValueError("model_config.flow_type must be a string")
    if flow_type not in {"nsf", "iaf"}:
        raise ValueError("model_config.flow_type must be 'nsf' or 'iaf'")

    iaf_kwargs = model_config["iaf_kwargs"]
    if not isinstance(iaf_kwargs, Mapping):
        raise ValueError("model_config.iaf_kwargs must be a mapping")
    if not all(isinstance(key, str) for key in iaf_kwargs):
        raise ValueError("model_config.iaf_kwargs keys must be strings")
    iaf_keys = set(iaf_kwargs)
    expected_iaf_keys = _IAF_CONFIG_KEYS if flow_type == "iaf" else set()
    if iaf_keys != expected_iaf_keys:
        difference = _describe_key_difference(iaf_keys, expected_iaf_keys)
        raise ValueError(f"Invalid model_config.iaf_kwargs schema ({difference})")
    for name, value in model_config.items():
        if name == "iaf_kwargs":
            for option_name, option_value in iaf_kwargs.items():
                _validate_plain_value(
                    option_value, f"model_config.iaf_kwargs.{option_name}"
                )
        else:
            _validate_plain_value(value, f"model_config.{name}")

    for name in ("dim", "num_blocks", "num_bins", "hidden_features"):
        value = model_config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"model_config.{name} must be a positive integer")
    for name in ("tail_bound", "max_derivative"):
        value = model_config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"model_config.{name} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"model_config.{name} must be finite")
    if model_config["tail_bound"] <= 0:
        raise ValueError("model_config.tail_bound must be strictly positive")
    if model_config["max_derivative"] <= 1:
        raise ValueError("model_config.max_derivative must be greater than 1")

    data_mean = model_config["data_mean"]
    data_std = model_config["data_std"]
    if (data_mean is None) != (data_std is None):
        raise ValueError("model_config.data_mean and data_std must both be set or None")
    _validate_data_statistic(
        data_mean,
        name="data_mean",
        dim=model_config["dim"],
        positive=False,
    )
    _validate_data_statistic(
        data_std,
        name="data_std",
        dim=model_config["dim"],
        positive=True,
    )

    if flow_type == "iaf":
        blocks_per_layer = iaf_kwargs["num_blocks_per_layer"]
        if (
            isinstance(blocks_per_layer, bool)
            or not isinstance(blocks_per_layer, int)
            or blocks_per_layer <= 0
        ):
            raise ValueError(
                "model_config.iaf_kwargs.num_blocks_per_layer must be a "
                "positive integer"
            )
        dropout_prob = iaf_kwargs["dropout_prob"]
        if isinstance(dropout_prob, bool) or not isinstance(
            dropout_prob, (int, float)
        ):
            raise ValueError("model_config.iaf_kwargs.dropout_prob must be numeric")
        if not math.isfinite(dropout_prob):
            raise ValueError("model_config.iaf_kwargs.dropout_prob must be finite")
        for name in _IAF_CONFIG_KEYS - {"num_blocks_per_layer", "dropout_prob"}:
            if not isinstance(iaf_kwargs[name], bool):
                raise ValueError(
                    f"model_config.iaf_kwargs.{name} must be a boolean"
                )

    return dict(model_config)


def _validate_model_matches_config(model, model_config):
    if model_config["flow_type"] != "nsf":
        return

    transforms = [
        transform
        for transform in model.flow._transform._transforms
        if isinstance(
            transform,
            BoundedPiecewiseRationalQuadraticAutoregressiveTransform,
        )
    ]
    if len(transforms) != model_config["num_blocks"]:
        raise ValueError("Model NSF topology has drifted from its checkpoint config")

    expected_floor = 1e-3 if model_config["dim"] == 1 else 1e-2
    for transform in transforms:
        if transform.max_derivative != model_config["max_derivative"]:
            raise ValueError(
                "Model max_derivative has drifted from its checkpoint config"
            )
        if transform.tail_bound != model_config["tail_bound"]:
            raise ValueError("Model tail_bound has drifted from its checkpoint config")
        if transform.num_bins != model_config["num_bins"]:
            raise ValueError("Model num_bins has drifted from its checkpoint config")
        for name in ("min_bin_width", "min_bin_height", "min_derivative"):
            if getattr(transform, name) != expected_floor:
                raise ValueError(
                    f"Model {name} has drifted from its supported checkpoint config"
                )


def _validate_state_dict(state_dict, model_dtype):
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping")
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("state_dict keys must be strings")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise ValueError("state_dict values must be tensors")

    floating_dtypes = {
        value.dtype for value in state_dict.values() if value.is_floating_point()
    }
    if len(floating_dtypes) != 1:
        raise ValueError("state_dict must use exactly one floating-point dtype")
    state_dtype = next(iter(floating_dtypes))
    if state_dtype not in _DTYPE_TO_NAME:
        raise ValueError(f"Unsupported checkpoint floating dtype: {state_dtype}")
    if _DTYPE_TO_NAME[state_dtype] != model_dtype:
        raise ValueError("model_dtype does not match the state_dict floating dtype")


def _validate_state_dict_metadata(state_dict, expected_state_dict):
    actual_keys = set(state_dict)
    expected_keys = set(expected_state_dict)
    if actual_keys != expected_keys:
        difference = _describe_key_difference(actual_keys, expected_keys)
        raise ValueError(f"state_dict keys do not match model_config ({difference})")

    for key, value in state_dict.items():
        expected = expected_state_dict[key]
        if value.shape != expected.shape:
            raise ValueError(f"state_dict tensor shape mismatch for {key!r}")
        if value.dtype != expected.dtype:
            raise ValueError(f"state_dict tensor dtype mismatch for {key!r}")
        if value.layout != expected.layout:
            raise ValueError(f"state_dict tensor layout mismatch for {key!r}")


def _model_dtype_name(state_dict):
    floating_dtypes = {
        value.dtype for value in state_dict.values() if value.is_floating_point()
    }
    if len(floating_dtypes) != 1:
        raise ValueError(
            "NormalizingFlowDensity must use exactly one floating-point dtype"
        )
    dtype = next(iter(floating_dtypes))
    if dtype not in _DTYPE_TO_NAME:
        raise ValueError(f"Unsupported model floating dtype: {dtype}")
    return _DTYPE_TO_NAME[dtype]


def save_normalizing_flow_checkpoint(model, destination):
    """Save a versioned model-only checkpoint using tensors and primitives."""
    if type(model) is not NormalizingFlowDensity:
        raise TypeError("model must be a NormalizingFlowDensity instance")

    model_config = model._get_checkpoint_model_config()
    _validate_model_config(model_config)
    _validate_model_matches_config(model, model_config)
    state_dict = model.state_dict()
    model_dtype = _model_dtype_name(state_dict)
    _validate_state_dict(state_dict, model_dtype)

    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "package_version": str(_PACKAGE_VERSION),
        "model_type": _MODEL_TYPE,
        "model_config": model_config,
        "model_dtype": model_dtype,
        "state_dict": state_dict,
    }
    torch.save(payload, destination)


def load_normalizing_flow_checkpoint(source, *, map_location="cpu"):
    """Reconstruct and strictly load a versioned model-only checkpoint."""
    payload = torch.load(
        source,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    if "format" not in payload:
        raise ValueError(
            "Not a versioned convMMD checkpoint. Reconstruct v0.1.0 models "
            "with their explicit configuration and load the raw state_dict "
            "with strict=True."
        )

    if not all(isinstance(key, str) for key in payload):
        raise ValueError("Checkpoint keys must be strings")
    payload_keys = set(payload)
    if payload_keys != _PAYLOAD_KEYS:
        difference = _describe_key_difference(payload_keys, _PAYLOAD_KEYS)
        raise ValueError(f"Invalid checkpoint schema ({difference})")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {payload['format']!r}")
    if (
        isinstance(payload["format_version"], bool)
        or not isinstance(payload["format_version"], int)
        or payload["format_version"] != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            f"Unsupported checkpoint format version: {payload['format_version']!r}"
        )
    if payload["model_type"] != _MODEL_TYPE:
        raise ValueError(f"Unsupported checkpoint model type: {payload['model_type']!r}")
    if not isinstance(payload["package_version"], str):
        raise ValueError("package_version must be a string")

    model_dtype = payload["model_dtype"]
    if not isinstance(model_dtype, str):
        raise ValueError("model_dtype must be a string")
    if model_dtype not in _NAME_TO_DTYPE:
        raise ValueError(f"Unsupported checkpoint model_dtype: {model_dtype!r}")
    model_config = _validate_model_config(payload["model_config"])
    _validate_state_dict(payload["state_dict"], model_dtype)

    constructor_config = dict(model_config)
    iaf_kwargs = constructor_config.pop("iaf_kwargs")
    try:
        model = NormalizingFlowDensity(**constructor_config, **iaf_kwargs)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint model_config cannot construct the model") from error
    model = model.to(dtype=_NAME_TO_DTYPE[model_dtype])
    _validate_state_dict_metadata(payload["state_dict"], model.state_dict())

    try:
        incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint state_dict is incompatible with its model_config"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Checkpoint state_dict failed strict loading")

    return model
