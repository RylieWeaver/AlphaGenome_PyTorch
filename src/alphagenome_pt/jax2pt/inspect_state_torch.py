"""
Inspect the full PyTorch AlphaGenome model and write param/state keys and shapes to text.
"""

from __future__ import annotations

# External
import argparse
from pathlib import Path

# Internal
from alphagenome_pt import deepmind_model
from .utils import summarize_params_and_state, write_summary



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect full PyTorch AlphaGenome model and write all keys/shapes to text"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for PyTorch model inspection. Defaults to CPU. Examples: cpu, gpu, cuda, mps.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).resolve().parent / "torch_params.txt",
        help="Output text file for key/shape summary.",
    )
    args = parser.parse_args()

    model = deepmind_model(device=args.device)
    params = dict(model.named_parameters())
    state_dict = model.state_dict()
    state = {
        key: value
        for key, value in state_dict.items()
        if key not in params
    }
    mapping = summarize_params_and_state(
        params=params,
        state=state,
    )
    write_summary(mapping, args.output)


if __name__ == "__main__":
    main()
