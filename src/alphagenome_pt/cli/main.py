"""Primary command-line entry point for alphagenome_pt."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

# Internal
from ..utils import package_version
from . import download


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        prog="alphagenome-pt",
        description="Command-line tools for AlphaGenome PyTorch.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version()}",
    )

    subparsers = parser.add_subparsers(dest="command")
    download_parser = subparsers.add_parser(
        "download",
        help=download.COMMAND_HELP,
        description=download.DESCRIPTION,
    )
    download.configure_parser(download_parser)
    download_parser.set_defaults(_handler=download.run)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the top-level command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


if __name__ == "__main__":
    main()
