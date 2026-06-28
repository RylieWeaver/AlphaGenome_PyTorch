from __future__ import annotations

# External
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch import Tensor



OUTPUT_HEADS = (
    "atac",
    "dnase",
    "procap",
    "cage",
    "rna_seq",
    "chip_tf",
    "chip_histone",
    "contact_maps",
    "splice_sites",
    "splice_site_usage",
    "splice_junctions",
)


def _metadata_table_to_json(table: Any) -> dict[str, Any]:
    if not hasattr(table, "to_json"):
        raise TypeError(
            "Expected a pandas-like metadata table that has a 'to_json' method, "
            f"got {type(table)!r}"
        )

    return {
        "num_rows": len(table),
        "columns": [str(column) for column in table.columns],
        "records": json.loads(table.to_json(orient="records")),
    }


def public_jax_metadata_to_json(
    loaded: list[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    from .load import load_public_metadata

    if loaded is None:
        loaded = load_public_metadata()
    return {
        "organisms": {
            organism_name: {
                head_name: _metadata_table_to_json(
                    get_output_metadata(metadata, head_name)
                )
                for head_name in OUTPUT_HEADS
            }
            for organism_name, metadata in loaded
        }
    }


def get_output_metadata(jax_metadata: Any, attr_name: str) -> Any:
    # NOTE: Can handle both jax_metadata[attr_name] and jax_metadata.attr_name
    if isinstance(jax_metadata, Mapping):
        if attr_name not in jax_metadata:
            raise AttributeError(f"JAX metadata is missing output metadata: {attr_name}")
        return jax_metadata[attr_name]
    if not hasattr(jax_metadata, attr_name):
        raise AttributeError(f"JAX metadata is missing output metadata: {attr_name}")
    return getattr(jax_metadata, attr_name)


def flatten_nested_dict(d: Mapping[str, Any], parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    """Flatten a nested pytree-like dict into path strings."""
    items: list[tuple[str, Any]] = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, Mapping):
            items.extend(flatten_nested_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def get_shape_and_dtype(arr: Any) -> dict[str, Any]:
    """Extract shape and dtype from a JAX/numpy/torch array."""
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def summarize_params_and_state(
    params: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return shape/dtype summaries for flat params and state mappings."""
    return {
        "params": {
            key: get_shape_and_dtype(arr)
            for key, arr in params.items()
        },
        "state": {
            key: get_shape_and_dtype(arr)
            for key, arr in state.items()
        },
    }


def summary_text(mapping: dict[str, Any]) -> str:
    """Return a human-readable summary of params/state keys, shapes, and dtypes."""
    lines = []
    lines.append("=" * 80)
    lines.append("PARAMETERS")
    lines.append("=" * 80)
    for key, info in sorted(mapping["params"].items()):
        shape_str = "x".join(str(s) for s in info["shape"])
        lines.append(f"  {key}: [{shape_str}] ({info['dtype']})")

    lines.append("")
    lines.append("=" * 80)
    lines.append("STATE")
    lines.append("=" * 80)
    for key, info in sorted(mapping["state"].items()):
        shape_str = "x".join(str(s) for s in info["shape"])
        lines.append(f"  {key}: [{shape_str}] ({info['dtype']})")

    n_params = len(mapping["params"])
    n_state = len(mapping["state"])
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"Total: {n_params} parameter arrays, {n_state} state arrays")
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def write_summary(mapping: dict[str, Any], output_path: Path) -> None:
    """Write a human-readable summary of params/state keys, shapes, and dtypes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary_text(mapping))


def print_summary(mapping: dict[str, Any]) -> None:
    """Print a human-readable summary of params/state keys, shapes, and dtypes."""
    print(summary_text(mapping), end="")


def jax_to_torch_tensor(arr: Any) -> Tensor:
    """Convert a JAX/numpy array into a detached torch tensor."""
    if isinstance(arr, Tensor):
        return arr.detach().clone()
    if hasattr(arr, "numpy"):
        np_arr = arr.numpy()
    else:
        np_arr = np.asarray(arr)
    return torch.from_numpy(np_arr.copy())


def shape_mismatches(
    converted_state_dict: Mapping[str, Tensor],
    reference_state_dict: Mapping[str, Tensor],
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return converted keys whose shape differs from the reference state_dict."""
    mismatches = {}
    for key, tensor in converted_state_dict.items():
        if key not in reference_state_dict:
            continue
        converted_shape = tuple(tensor.shape)
        reference_shape = tuple(reference_state_dict[key].shape)
        if converted_shape != reference_shape:
            mismatches[key] = (converted_shape, reference_shape)
    return mismatches


def check_converted_shapes(
    converted_state_dict: Mapping[str, Tensor],
    reference_state_dict: Mapping[str, Tensor],
) -> None:
    """Raise if any converted tensor has the wrong shape for the reference state_dict."""
    mismatches = shape_mismatches(converted_state_dict, reference_state_dict)
    if mismatches:
        lines = "\n".join(
            f"{key}: converted={converted_shape}, reference={reference_shape}"
            for key, (converted_shape, reference_shape) in sorted(mismatches.items())
        )
        raise ValueError(f"Converted tensor shape mismatches ({len(mismatches)}):\n{lines}")
