# General
import sys
from pathlib import Path
from typing import Any

# Torch
import torch

# Paths
JAX2PT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = JAX2PT_ROOT.parents[1]
if str(JAX2PT_ROOT) not in sys.path:
    sys.path.insert(0, str(JAX2PT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inspect_jax import load_jax_checkpoint
from mapping import (
    convert_flat_jax_to_torch,
    expected_torch_only_param_keys,
    expected_torch_only_keys,
    expected_torch_only_state_keys,
    key_differences,
    mapping_differences_report,
    mapped_param_keys,
    mapped_state_keys,
)
from utils import check_converted_shapes, flatten_nested_dict, full_alphagenome_model

# Commands
# pytest -s test_mappings.py


def flat_jax_checkpoint() -> tuple[dict[str, Any], dict[str, Any]]:
    print("Loading official JAX checkpoint...")
    params, state = load_jax_checkpoint(use_gpu=False)
    print("Flattening official JAX checkpoint...")
    flat_params = flatten_nested_dict(params)
    flat_state = flatten_nested_dict(state)
    print(f"Loaded JAX checkpoint: {len(flat_params)} params, {len(flat_state)} state tensors")
    return flat_params, flat_state


def flat_torch_checkpoint(model) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    params = dict(model.named_parameters())
    state = dict(model.named_buffers())
    return params, state


def mapped_jax_checkpoint() -> tuple[dict[str, None], dict[str, None]]:
    mapped_params, _ = mapped_param_keys()
    mapped_state, _ = mapped_state_keys()
    return dict.fromkeys(mapped_params), dict.fromkeys(mapped_state)


def mapped_torch_checkpoint() -> tuple[dict[str, None], dict[str, None]]:
    _, mapped_params = mapped_param_keys()
    _, mapped_state = mapped_state_keys()
    return dict.fromkeys(mapped_params), dict.fromkeys(mapped_state)


def assert_no_key_differences(
    left_params: dict[str, Any],
    left_state: dict[str, Any],
    right_params: dict[str, Any],
    right_state: dict[str, Any],
    *,
    expected_left_only_params: set[str] | None = None,
    expected_left_only_state: set[str] | None = None,
    expected_right_only_params: set[str] | None = None,
    expected_right_only_state: set[str] | None = None,
) -> None:
    diffs = key_differences(left_params, left_state, right_params, right_state)
    diffs["left_only_params"] -= expected_left_only_params or set()
    diffs["left_only_state"] -= expected_left_only_state or set()
    diffs["right_only_params"] -= expected_right_only_params or set()
    diffs["right_only_state"] -= expected_right_only_state or set()
    assert all(not keys for keys in diffs.values())


# NOTE: Any function with "test_" prefix will be automatically run by pytest
def test_official_jax_checkpoint_maps_to_torch_model():
    torch_model = full_alphagenome_model()
    jax_params, jax_state = flat_jax_checkpoint()
    torch_params, torch_state = flat_torch_checkpoint(torch_model)
    mapped_jax_params, mapped_jax_state = mapped_jax_checkpoint()
    mapped_torch_params, mapped_torch_state = mapped_torch_checkpoint()
    report = mapping_differences_report(
        jax_params,
        jax_state,
        torch_params,
        torch_state,
    )
    print(report)

    assert_no_key_differences(
        jax_params,
        jax_state,
        mapped_jax_params,
        mapped_jax_state,
    )
    assert_no_key_differences(
        torch_params,
        torch_state,
        mapped_torch_params,
        mapped_torch_state,
        expected_left_only_params=expected_torch_only_param_keys(torch_params),
        expected_left_only_state=expected_torch_only_state_keys(torch_state),
    )

    converted = convert_flat_jax_to_torch(jax_params, jax_state)
    torch_checkpoint = torch_model.state_dict()
    check_converted_shapes(converted, torch_checkpoint)

    load_result = torch_model.load_state_dict(converted, strict=False, assign=True)
    unexpected = set(load_result.unexpected_keys)
    expected_missing = expected_torch_only_keys(
        {},
        {key: torch_checkpoint[key] for key in load_result.missing_keys},
    )
    missing = set(load_result.missing_keys) - expected_missing
    print(f"Unexpected keys when loading converted checkpoint: {unexpected}")
    print(f"Missing keys when loading converted checkpoint: {missing}")

    assert unexpected == set()
    assert missing == set()
