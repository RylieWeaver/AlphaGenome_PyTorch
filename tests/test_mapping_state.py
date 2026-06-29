# External
import os
from typing import Any

import pytest
import torch

# Internal
from alphagenome_pt import deepmind_model
from alphagenome_pt.jax2pt.load import load_jax_state
from alphagenome_pt.jax2pt.mapping_state import (
    convert_flat_jax_to_torch,
    expected_torch_only_param_keys,
    expected_torch_only_keys,
    expected_torch_only_state_keys,
    key_differences,
    mapping_differences_report,
    mapped_param_keys,
    mapped_state_keys,
)
from alphagenome_pt.jax2pt.utils import check_converted_shapes, flatten_nested_dict



def flat_jax_checkpoint() -> tuple[dict[str, Any], dict[str, Any]]:
    print("Loading official JAX checkpoint...")
    params, state = load_jax_state(device="cpu")
    print("Flattening official JAX checkpoint...")
    flat_params = flatten_nested_dict(params)
    flat_state = flatten_nested_dict(state)
    print(f"Loaded JAX checkpoint: {len(flat_params)} params, {len(flat_state)} state tensors")
    return flat_params, flat_state


def flat_torch_checkpoint(model) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    params = dict(model.named_parameters())
    state_dict = model.state_dict()
    state = {
        key: value
        for key, value in state_dict.items()
        if key not in params
    }
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


@pytest.mark.skipif(
    os.environ.get("ALPHAGENOME_PT_RUN_JAX_MAPPING_TEST") != "1",
    reason="Set ALPHAGENOME_PT_RUN_JAX_MAPPING_TEST=1 to download and compare the official JAX checkpoint.",
)
def test_official_jax_checkpoint_maps_to_torch_model():
    torch_model = deepmind_model()
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
    missing_params = {
        key: torch_params[key]
        for key in load_result.missing_keys
        if key in torch_params
    }
    missing_state = {
        key: torch_state[key]
        for key in load_result.missing_keys
        if key in torch_state
    }
    expected_missing = expected_torch_only_keys(
        missing_params,
        missing_state,
    )
    missing = set(load_result.missing_keys) - expected_missing
    print(f"Unexpected keys when loading converted checkpoint: {unexpected}")
    print(f"Missing keys when loading converted checkpoint: {missing}")

    assert unexpected == set()
    assert missing == set()
