"""Convert public AlphaGenome JAX metadata into alphagenome_pt metadata JSON."""

from __future__ import annotations

# External
import argparse
import json
from pathlib import Path

# Internal
from alphagenome_pt import (
    DEFAULT_CONVERTED_METADATA_FILENAME,
    DEFAULT_METADATA_SUMMARY_FILENAME,
    DEFAULT_RAW_METADATA_FILENAME,
)
from .load import load_public_metadata
from .mapping_metadata import (
    convert_metadata,
    summarize_metadata,
)
from .utils import public_jax_metadata_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(DEFAULT_CONVERTED_METADATA_FILENAME),
        help="JSON output path for converted metadata.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path(DEFAULT_RAW_METADATA_FILENAME),
        help="JSON output path for raw public AlphaGenome metadata.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(DEFAULT_METADATA_SUMMARY_FILENAME),
        help="JSON output path for compact metadata summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading public JAX metadata")
    public_metadata = load_public_metadata()

    print("Saving raw public JAX metadata to JSON")
    raw_metadata = public_jax_metadata_to_json(public_metadata)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps(raw_metadata, indent=2) + "\n")
    print(f"Wrote raw metadata JSON to {args.raw_output}")

    print("Converting to alphagenome_pt metadata")
    converted_metadata = convert_metadata(public_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted_metadata, indent=2) + "\n")
    print(f"Completed conversion of metadata to JSON at {args.output}")

    print("Summarizing metadata")
    summary_metadata = summarize_metadata(converted_metadata)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary_metadata, indent=2) + "\n")
    print(f"Wrote metadata summary JSON to {args.summary_output}")


if __name__ == "__main__":
    main()
