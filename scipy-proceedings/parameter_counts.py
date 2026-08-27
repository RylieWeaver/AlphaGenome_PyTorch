#!/usr/bin/env python3
"""Measure published and from-scratch AlphaGenome parameter counts."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
from alphagenome_pt import AlphaGenome, AlphaGenomeConfig, Metadata, deepmind_model


DEFAULT_NUM_CHANNELS_GRID = (64, 128, 256, 512, 768)
DEFAULT_MAX_SEQ_LEN = 1_048_576
FIELDNAMES = (
    "model",
    "num_channels",
    "max_seq_len",
    "num_organisms",
    "num_heads",
    "total_parameters",
    "trainable_parameters",
    "parameter_bytes",
    "parameter_gib",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_CHANNELS_GRID,
        help="Base channel counts for the from-scratch models.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Maximum sequence length for models (default: 1,048,576).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent / "results" / "parameter-counts.csv"
        ),
    )
    return parser.parse_args()


def describe_model(name: str, model: AlphaGenome) -> dict[str, object]:
    parameters = tuple(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    return {
        "model": name,
        "num_channels": model.num_channels,
        "max_seq_len": model.max_seq_len,
        "num_organisms": model.num_organisms,
        "num_heads": len(model._heads),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "parameter_bytes": parameter_bytes,
        "parameter_gib": f"{parameter_bytes / 2**30:.6f}",
    }


def single_organism_headless_metadata() -> Metadata:
    return Metadata({"organisms": ["organism_0"], "heads": {}})


def validate_args(args: argparse.Namespace) -> None:
    if args.max_seq_len < 2_048 or args.max_seq_len % 2_048:
        raise ValueError("--max-seq-len must be at least 2,048 and divisible by 2,048.")
    if any(channels <= 0 for channels in args.channels):
        raise ValueError("--channels values must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    rows: list[dict[str, object]] = []

    published = deepmind_model(device="cpu", load_state=False)
    rows.append(describe_model("published_deepmind", published))
    del published
    gc.collect()

    for channels in args.channels:
        model = AlphaGenome(
            AlphaGenomeConfig(
                max_seq_len=args.max_seq_len,
                num_channels=channels,
                metadata=single_organism_headless_metadata(),
            )
        )
        rows.append(
            describe_model(f"from_scratch_headless_c{channels}", model)
        )
        del model
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FIELDNAMES).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
