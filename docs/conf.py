"""Sphinx configuration for the AlphaGenome PyTorch documentation."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


project = "AlphaGenome PyTorch"
author = "Rylie Weaver, Gomathi Lakshmanan, and John Lagergren"
copyright = "2026, AlphaGenome PyTorch contributors"

project_root = Path(__file__).resolve().parents[1]
with (project_root / "pyproject.toml").open("rb") as file:
    release = tomllib.load(file)["project"]["version"]

version = release

extensions = ["myst_parser", "sphinx_design", "sphinx_copybutton"]

source_suffix = {
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]
myst_heading_anchors = 3

# Add copy controls only to runnable examples, not literal output or tree dumps.
copybutton_selector = (
    "div.highlight-python div.highlight pre, "
    "div.highlight-bash div.highlight pre"
)

html_theme = "sphinx_book_theme"
html_title = "AlphaGenome PyTorch Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/RylieWeaver/AlphaGenome_PyTorch",
    "use_repository_button": True,
    "use_issues_button": True,
    "show_toc_level": 2,
}
