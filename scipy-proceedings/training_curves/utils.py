"""Shared utilities for the proceedings training-curve scripts."""

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).with_name("downloads")
DEFAULT_FASTA = DATA_DIR / "hg38-chr1.fa"
DEFAULT_BIGWIG = DATA_DIR / "ENCFF877MMK.bigWig"


def output_dir(args, task: str) -> Path:
    """Return the requested or default task result directory."""

    if args.output_dir is not None:
        return args.output_dir
    initialization = "checkpoint" if args.load_state else "from-scratch"
    return (
        Path(__file__).resolve().parents[1]
        / "results"
        / f"{task}-training-{initialization}"
    )


def save_results(
    output_dir,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    """Write step metrics and run-level metadata to one result directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


def validate_args(args) -> None:
    """Validate arguments shared by the training-curve scripts."""

    if args.sequence_length < 2_048 or args.sequence_length % 2_048:
        raise ValueError(
            "--sequence-length must be at least 2,048 and divisible by 2,048."
        )
    if args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("--batch-size and --steps must be positive.")
    if args.eval_every <= 0 or args.eval_batches <= 0:
        raise ValueError("--eval-every and --eval-batches must be positive.")
