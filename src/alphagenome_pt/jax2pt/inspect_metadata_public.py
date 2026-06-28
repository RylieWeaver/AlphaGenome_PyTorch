"""Download public AlphaGenome JAX metadata and save it as JSON.

Run:
    python -m alphagenome_pt.jax2pt.inspect_metadata_public
    python -m alphagenome_pt.jax2pt.inspect_metadata_public --output alphagenome_metadata_raw.json
"""

from __future__ import annotations

# External
import argparse
import json
from pathlib import Path

# Internal
from alphagenome_pt import DEFAULT_RAW_METADATA_FILENAME
from .utils import public_jax_metadata_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(DEFAULT_RAW_METADATA_FILENAME),
        help="JSON output path for raw public AlphaGenome JAX metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = public_jax_metadata_to_json()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
