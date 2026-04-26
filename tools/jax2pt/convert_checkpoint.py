"""Convert the official JAX AlphaGenome checkpoint to a PyTorch state_dict."""

from __future__ import annotations

# General
import argparse
from pathlib import Path

# Torch
import torch

# JAX2PT
from inspect_jax import load_jax_checkpoint
from mapping import convert_flat_jax_to_torch
from utils import flatten_nested_dict



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-version",
        default="all_folds",
        help="Official AlphaGenome checkpoint version to load.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "alphagenome_converted_state_dict.pt",
        help="Path where the converted PyTorch state_dict will be written.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use JAX GPU loading if available. Defaults to CPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading JAX checkpoint: {args.model_version}")
    params, state = load_jax_checkpoint(
        model_version=args.model_version,
        use_gpu=args.gpu,
    )

    print("Flattening JAX params/state")
    flat_params = flatten_nested_dict(params)
    flat_state = flatten_nested_dict(state)

    print("Converting to PyTorch state_dict")
    converted = convert_flat_jax_to_torch(flat_params, flat_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)

    print(f"Saved {len(converted)} tensors to {args.output}")


if __name__ == "__main__":
    main()
