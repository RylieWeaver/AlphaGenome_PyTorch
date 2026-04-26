"""
Inspect JAX/Haiku checkpoint and write all param/state keys and shapes to text.

Environment variables:
    HF_TOKEN: HuggingFace access token for gated model access
    HF_HOME: HuggingFace cache directory (default: ~/.cache/huggingface)
"""
from __future__ import annotations

# General
import argparse
import os
import sys
from pathlib import Path
from typing import Any

# JAX2PT
from utils import flatten_nested_dict, get_shape_and_dtype, write_summary


# NOTE:
# - uv pip install -e /Users/r9w/Coding/Scratch/alphagenome_research


def load_jax_checkpoint(model_version: str = "all_folds", use_gpu: bool = True, verbose: bool = True):
    """Load JAX checkpoint using official alphagenome_research loader.

    Args:
        model_version: Model version to load (e.g., "all_folds", "fold_0", etc.)
        use_gpu: Whether to use GPU if available (default: True)

    Returns:
        Tuple of (params, state) where each is a nested dict of JAX arrays.
    """
    # Lazy import to avoid requiring JAX deps at module load time
    try:
        if verbose:
            print("Importing JAX...", flush=True)
        import jax
        if verbose:
            print("Importing alphagenome_research loader...", flush=True)
        from alphagenome_research.model.dna_model import create, create_from_huggingface
        if verbose:
            print("JAX loader imports complete.", flush=True)
    except ImportError as e:
        print(f"Error: Missing required dependencies. Install with: pip install -e '.[convert]'")
        print(f"Details: {e}")
        sys.exit(1)

    # Select device
    if use_gpu:
        try:
            device = jax.devices("gpu")[0]
            if verbose:
                print(f"Using GPU device: {device}")
        except RuntimeError:
            device = jax.devices("cpu")[0]
            if verbose:
                print(f"No GPU available, using CPU device: {device}")
    else:
        device = jax.devices("cpu")[0]
        if verbose:
            print(f"Using CPU device: {device}")

    local_model_path = os.environ.get("ALPHAGENOME_MODEL_PATH")
    if local_model_path and Path(local_model_path).exists():
        if verbose:
            print(f"Loading checkpoint from local path: {local_model_path}")
            print("Calling alphagenome_research create(...)", flush=True)
        model = create(
            checkpoint_path=local_model_path,
            device=device,
        )
        if verbose:
            print("alphagenome_research create(...) returned.", flush=True)
    else:
        if verbose:
            print(f"Loading checkpoint (model_version={model_version})...")
            print("Calling alphagenome_research create_from_huggingface(...)", flush=True)
        model = create_from_huggingface(
            model_version=model_version,
            device=device,
        )
        if verbose:
            print("alphagenome_research create_from_huggingface(...) returned.", flush=True)

    # Extract params and state from the model (note: underscore prefix)
    params = model._params
    state = model._state

    return params, state


def inspect_checkpoint(params, state) -> dict[str, Any]:
    """Generate a mapping of all parameter and state keys with their shapes."""
    result = {
        "params": {},
        "state": {},
    }

    # Flatten and extract shapes for params
    flat_params = flatten_nested_dict(params)
    for key, arr in flat_params.items():
        result["params"][key] = get_shape_and_dtype(arr)

    # Flatten and extract shapes for state
    flat_state = flatten_nested_dict(state)
    for key, arr in flat_state.items():
        result["state"][key] = get_shape_and_dtype(arr)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Inspect JAX/Haiku checkpoint and write all keys/shapes to text"
    )
    parser.add_argument(
        "--model-version",
        default="all_folds",
        help="Model version to load (e.g., 'all_folds', 'fold_0')",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU device (default: use GPU if available)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).resolve().parent / "jax_params.txt",
        help="Output text file for key/shape summary.",
    )
    args = parser.parse_args()

    params, state = load_jax_checkpoint(
        args.model_version,
        use_gpu=not args.cpu,
        verbose=False,
    )
    mapping = inspect_checkpoint(params, state)
    write_summary(mapping, args.output)


if __name__ == "__main__":
    main()
