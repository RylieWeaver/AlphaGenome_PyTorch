# External
import argparse
from pathlib import Path

# Internal
from ..checkpoint import (
    DEFAULT_ALPHAGENOME_REPO_ID,
    download_deepmind_metadata,
    download_deepmind_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download converted DeepMind AlphaGenome PyTorch files from Hugging Face."
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("."),
        help="Directory where checkpoint and metadata files are copied.",
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
        "--token",
        default=None,
        help="Optional Hugging Face auth token.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force a fresh download from Hugging Face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = args.local_dir.expanduser()

    metadata_path = download_deepmind_metadata(
        local_dir,
        repo_id=args.repo_id,
        repo_dir=args.repo_dir,
        token=args.token,
        force_download=args.force_download,
    )
    state_paths = download_deepmind_state(
        local_dir,
        download_all_folds=True,
        repo_id=args.repo_id,
        repo_dir=args.repo_dir,
        token=args.token,
        force_download=args.force_download,
    )

    print(f"Downloaded metadata: {metadata_path}")
    for state_path in state_paths:
        print(f"Downloaded state: {state_path}")


if __name__ == "__main__":
    main()
