from __future__ import annotations

# General
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import sys

# Torch
import torch
from torch import Tensor

# JAX2PT

JAX2PT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = JAX2PT_ROOT.parents[1]


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


def full_alphagenome_metadata() -> dict:
    def head(num_tracks: int) -> dict:
        return {
            "num_tracks": [num_tracks, num_tracks],
            "means": [[1.0] * num_tracks, [1.0] * num_tracks],
        }

    return {
        "organisms": ["human", "mouse"],
        "heads": {
            "atac": head(256),
            "dnase": head(384),
            "procap": head(128),
            "cage": head(640),
            "rna_seq": head(768),
            "chip_tf": head(1664),
            "chip_histone": head(1152),
            "contact_maps": head(28),
            "splice_sites_classification": head(5),
            "splice_sites_usage": head(734),
            "splice_sites_junction": {
                "num_tissues": [367, 367],
                "means": [[1.0] * 367, [1.0] * 367],
            },
        },
    }


def full_alphagenome_config():
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from alphagenome_pt.model import AlphaGenomeConfig

    return AlphaGenomeConfig(
        max_seq_len=1_048_576,
        num_channels=768,
        channel_increment=128,
        transformer_layers=9,
        num_q_heads=8,
        num_kv_heads=1,
        qk_head_dim=128,
        v_head_dim=192,
        pair_channels=128,
        pair_heads=32,
        pos_channels=64,
        transformer_mlp_ratio=2,
        embedder_mlp_ratio=2,
        num_splice_sites=512,
        splice_site_channels=768,
        metadata=full_alphagenome_metadata(),
    )


def full_alphagenome_model():
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from alphagenome_pt.model import AlphaGenome

    return AlphaGenome(full_alphagenome_config())


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
