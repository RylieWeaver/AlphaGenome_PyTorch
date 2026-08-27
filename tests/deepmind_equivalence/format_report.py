# External
import argparse
from pathlib import Path
import pandas as pd


METRIC_FORMATS = {
    "relative_L2": lambda value: f"{value:.3e}",
    "relative_Linf": lambda value: f"{value:.3e}",
    "max_abs": lambda value: f"{value:.3e}",
    "mean_abs": lambda value: f"{value:.3e}",
    "reference_max_abs": lambda value: f"{value:.3e}",
    "reference_mean_abs": lambda value: f"{value:.3e}",
    "exact_fraction": lambda value: f"{value:.2%}",
    "num_values": lambda value: f"{value:,.0f}",
}


def write_markdown_report(report, path):
    """Write an equivalence report as a readable Markdown table."""
    display = report.copy()
    for column, formatter in METRIC_FORMATS.items():
        if column in display:
            display[column] = display[column].map(formatter)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = "# JAX-PyTorch Equivalence Report\n\n"
    path.write_text(
        markdown + display.to_markdown(index=False) + "\n", encoding="utf-8"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="equivalence report CSV")
    parser.add_argument(
        "output", type=Path, nargs="?", help="output Markdown path"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or args.report.with_suffix(".md")
    write_markdown_report(pd.read_csv(args.report), output)


if __name__ == "__main__":
    main()
