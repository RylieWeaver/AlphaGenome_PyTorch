"""General package utilities."""

from __future__ import annotations

# External
from importlib import metadata as importlib_metadata
from pathlib import Path
import tomllib



def project_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.exists():
            return parent
    return None


def project_metadata() -> dict | None:
    root = project_root()
    if root is None:
        return None

    with (root / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def package_name() -> str:
    project = project_metadata()
    if project is not None:
        return project["name"]
    return Path(__file__).resolve().parent.name


def package_version() -> str:
    name = package_name()
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        project = project_metadata()
        if project is None:
            raise RuntimeError(
                "Could not find installed package metadata or pyproject.toml."
            )
        return project["version"]
