"""Convert the official JAX AlphaGenome checkpoint to a PyTorch state_dict."""

from __future__ import annotations

# External
import argparse
from pathlib import Path
import torch

# Internal
from alphagenome_pt import FOLD_NAMES, fold_filename
from .load import load_jax_state
from .mapping_state import convert_state



def folds_to_convert(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all_checkpoints or args.fold is None:
        return FOLD_NAMES
    return (args.fold,)


def convert_fold(fold: str, *, model_path: Path | None, torch_output_dir: Path, device: str) -> Path:
    output = torch_output_dir / fold_filename(fold)

    print(f"Loading JAX checkpoint fold: {fold}")
    params, state = load_jax_state(
        fold=fold,
        model_path=model_path,
        device=device,
    )

    print("Converting to PyTorch state_dict")
    converted = convert_state(params, state)

    print(f"Saving tensors to {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)

    print(f"Completed conversion of fold {fold} to PyTorch state_dict at {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--fold",
        choices=FOLD_NAMES,
        default=None,
        help="Official AlphaGenome checkpoint fold to convert.",
    )
    checkpoint_group.add_argument(
        "--all-checkpoints",
        action="store_true",
        help=(
            "Convert every supported AlphaGenome checkpoint. "
            "This is the default when --fold is not set."
        ),
    )
    parser.add_argument(
        "--jax-input-dir",
        type=Path,
        default=None,
        help=(
            "Local official JAX checkpoint directory. If set, it must contain "
            "checkpoint files named alphagenome_{fold_name}.pt."
        ),
    )
    parser.add_argument(
        "--torch-output-dir",
        type=Path,
        default=Path("."),
        help=(
            "Directory where the converted PyTorch state_dict will be written. "
            "Output filename is alphagenome_{fold_name}.pt."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for JAX checkpoint loading. Defaults to CPU. Examples: cpu, gpu, cuda, mps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_output_dir = args.torch_output_dir.expanduser().resolve()
    jax_input_dir = None
    if args.jax_input_dir is not None:
        jax_input_dir = args.jax_input_dir.expanduser().resolve()
    folds = folds_to_convert(args)
    convert_all = len(folds) > 1

    outputs = []
    for fold in folds:
        model_path = None if jax_input_dir is None else jax_input_dir / fold_filename(fold)
        outputs.append(
            convert_fold(
                fold,
                model_path=model_path,
                torch_output_dir=torch_output_dir,
                device=args.device,
            )
        )

    if convert_all:
        print(f"Converted {len(outputs)} folds into {torch_output_dir}")


if __name__ == "__main__":
    main()
