"""General package utilities."""

from __future__ import annotations

# External
from importlib import metadata as importlib_metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


PACKAGE_DISTRIBUTION_NAME = "alphagenome-pt"


def _normalize_distribution_name(name: str) -> str:
    return name.replace("_", "-").lower()


def project_root() -> Path | None:
    # NOTE: This is hardcoded to the package layout:
    #   repo_root/src/alphagenome_pt/utils.py
    # We don't dynamically search because installed packages often live under
    # another uv/project directory that can have an unrelated pyproject.toml.
    root = Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return None

    with pyproject.open("rb") as f:
        project = tomllib.load(f).get("project", {})
    if (
        _normalize_distribution_name(project.get("name", ""))
        != PACKAGE_DISTRIBUTION_NAME
    ):
        return None
    return root


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
    return PACKAGE_DISTRIBUTION_NAME


def package_version() -> str:
    project = project_metadata()
    if project is not None:
        return project["version"]
    try:
        return importlib_metadata.version(PACKAGE_DISTRIBUTION_NAME)
    except importlib_metadata.PackageNotFoundError:
        raise RuntimeError(
            "Could not find installed package metadata or pyproject.toml."
        )
