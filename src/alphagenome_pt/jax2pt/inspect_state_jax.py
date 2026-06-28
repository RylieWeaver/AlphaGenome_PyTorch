"""
Inspect JAX/Haiku checkpoint and write all param/state keys and shapes to text.

Environment variables:
    HF_TOKEN: HuggingFace access token for gated model access
    HF_HOME: HuggingFace cache directory (default: ~/.cache/huggingface)
"""

from __future__ import annotations

# External
import argparse
from pathlib import Path

# Internal
from alphagenome_pt import DEFAULT_FOLD, FOLD_NAMES
from .load import load_jax_state
from .utils import flatten_nested_dict, summarize_params_and_state, write_summary



def main():
    parser = argparse.ArgumentParser(
        description="Inspect JAX/Haiku checkpoint and write all keys/shapes to text"
    )
    parser.add_argument(
        "--fold",
        choices=FOLD_NAMES,
        default=DEFAULT_FOLD,
        help="Checkpoint fold to load.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional official JAX checkpoint path. If set, this is used instead of --fold.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for JAX checkpoint loading. Defaults to CPU. Examples: cpu, gpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).resolve().parent / "jax_params.txt",
        help="Output text file for key/shape summary.",
    )
    args = parser.parse_args()

    params, state = load_jax_state(
        fold=args.fold,
        model_path=args.model_path,
        device=args.device,
        verbose=False,
    )
    mapping = summarize_params_and_state(
        params=flatten_nested_dict(params),
        state=flatten_nested_dict(state),
    )
    write_summary(mapping, args.output)


if __name__ == "__main__":
    main()
