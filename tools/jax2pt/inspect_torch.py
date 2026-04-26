"""
Inspect the full PyTorch AlphaGenome model and write param/state keys and shapes to text.
"""

from __future__ import annotations

# General
import argparse
from pathlib import Path
from typing import Any

# Torch
import torch

# JAX2PT
from utils import full_alphagenome_model, get_shape_and_dtype, write_summary



def inspect_model(model: torch.nn.Module) -> dict[str, Any]:
    """Generate a mapping of all parameter and state keys with their shapes."""
    result = {
        "params": {},
        "state": {},
    }

    for key, tensor in model.named_parameters():
        result["params"][key] = get_shape_and_dtype(tensor)

    for key, tensor in model.named_buffers():
        result["state"][key] = get_shape_and_dtype(tensor)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect full PyTorch AlphaGenome model and write all keys/shapes to text"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).resolve().parent / "torch_params.txt",
        help="Output text file for key/shape summary.",
    )
    args = parser.parse_args()

    model = full_alphagenome_model()
    mapping = inspect_model(model)
    write_summary(mapping, args.output)


if __name__ == "__main__":
    main()
