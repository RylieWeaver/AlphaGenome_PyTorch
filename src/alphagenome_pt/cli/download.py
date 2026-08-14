from __future__ import annotations

# NOTE: Delegate authentication to huggingface_hub, which uses saved credentials
# or HF_TOKEN. We don't accept tokens as CLI arguments because shell history and
# process listings can expose them.

import argparse
from collections.abc import Sequence
from pathlib import Path

# Internal
from ..checkpoint import (
    DEFAULT_ALPHAGENOME_REPO_ID,
    DEFAULT_FOLD,
    FOLD_NAMES,
    download_deepmind_metadata,
    download_deepmind_state,
)


COMMAND_HELP = "Download converted checkpoints and metadata."
DESCRIPTION = (
    "Download converted DeepMind AlphaGenome PyTorch files from Hugging Face."
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add download options to an AlphaGenome CLI parser."""
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("."),
        help="Directory where checkpoint and metadata files are copied.",
    )
    parser.add_argument(
        "--fold",
        choices=FOLD_NAMES,
        default=None,
        help="Checkpoint fold to download. Omit to download all folds.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_ALPHAGENOME_REPO_ID,
        help="Hugging Face Hub repo ID.",
    )
    parser.add_argument(
        "--repo-dir",
        default=None,
        help="Directory inside the Hub repo. Defaults to v{package-version}.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force a fresh download from Hugging Face.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Allow for single standalone download parser."""
    parser = argparse.ArgumentParser(
        prog="alphagenome-pt-download",
        description=DESCRIPTION,
    )
    configure_parser(parser)
    return parser


def run(args: argparse.Namespace) -> None:
    """Download the artifacts selected by parsed CLI arguments."""
    local_dir = args.local_dir.expanduser()

    metadata_path = download_deepmind_metadata(
        local_dir,
        repo_id=args.repo_id,
        repo_dir=args.repo_dir,
        force_download=args.force_download,
    )
    state_paths = download_deepmind_state(
        local_dir,
        fold=args.fold or DEFAULT_FOLD,
        download_all_folds=args.fold is None,
        repo_id=args.repo_id,
        repo_dir=args.repo_dir,
        force_download=args.force_download,
    )
    if not isinstance(state_paths, list):
        state_paths = [state_paths]

    print(f"Downloaded metadata: {metadata_path}")
    for state_path in state_paths:
        print(f"Downloaded state: {state_path}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the standalone download command."""
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
