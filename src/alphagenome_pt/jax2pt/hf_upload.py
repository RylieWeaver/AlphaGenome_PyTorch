"""Upload converted AlphaGenome checkpoints to Hugging Face Hub."""

from __future__ import annotations

# External
import argparse
from pathlib import Path

from huggingface_hub import HfApi

# Internal
from alphagenome_pt import (
    DEFAULT_ALPHAGENOME_REPO_ID,
    DEFAULT_CONVERTED_METADATA_FILENAME,
    DEFAULT_METADATA_SUMMARY_FILENAME,
    DEFAULT_RAW_METADATA_FILENAME,
    FOLD_NAMES,
    fold_filename,
)
from alphagenome_pt.utils import package_version


def checkpoint_paths(local_dir: Path) -> list[Path]:
    paths = [local_dir / fold_filename(fold_name) for fold_name in FOLD_NAMES]
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing expected converted checkpoints:\n"
            f"{missing_names}"
        )
    return paths


def metadata_paths(local_dir: Path) -> list[Path]:
    paths = [
        local_dir / DEFAULT_CONVERTED_METADATA_FILENAME,
        local_dir / DEFAULT_RAW_METADATA_FILENAME,
        local_dir / DEFAULT_METADATA_SUMMARY_FILENAME,
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing expected metadata files:\n"
            f"{missing_names}"
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("."),
        help="Local directory containing converted AlphaGenome checkpoint and metadata files.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_ALPHAGENOME_REPO_ID,
        help="Hugging Face model repo to upload to.",
    )
    parser.add_argument(
        "--repo-dir",
        default=None,
        help="Directory inside the Hub repo. Defaults to v{package-version}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print upload targets without creating a repo or uploading files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = args.local_dir.expanduser().resolve()
    repo_dir = args.repo_dir or f"v{package_version()}"
    paths = [
        *checkpoint_paths(local_dir),
        *metadata_paths(local_dir),
    ]

    if not paths:
        raise FileNotFoundError(f"No expected files found in {local_dir}.")

    for path in paths:
        path_in_repo = f"{repo_dir}/{path.name}"
        print(f"{path} -> hf://{args.repo_id}/{path_in_repo}")

    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    for path in paths:
        path_in_repo = f"{repo_dir}/{path.name}"
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type="model",
        )

    print(
        f"Uploaded {len(paths)} file(s) to "
        f"https://huggingface.co/{args.repo_id}/tree/main/{repo_dir}"
    )


if __name__ == "__main__":
    main()
