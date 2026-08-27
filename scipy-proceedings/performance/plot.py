#!/usr/bin/env python3
"""Plot latency and peak GPU memory from the performance CSV reports."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODES = {
    "embeddings": {"label": "Embeddings", "color": "#1f77b4", "linestyle": "-"},
    "predictions": {"label": "Predictions", "color": "#ff7f0e", "linestyle": "--"},
}


def format_length(length: float) -> str:
    return f"{length / 1_024:g}k"


def plot_benchmark(benchmark: str) -> None:
    reports = {}
    for mode in MODES:
        path = RESULTS_DIR / f"{benchmark}-{mode}.csv"
        if path.exists():
            report = pd.read_csv(path)
            reports[mode] = report[report["status"] == "ok"]

    if not reports:
        print(f"Skipping {benchmark}: no result CSVs found")
        return

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.25))
    figure.subplots_adjust(top=0.76, wspace=0.28)
    for mode, report in reports.items():
        style = MODES[mode]
        shared = {
            "marker": "o",
            "linewidth": 2,
            "color": style["color"],
            "linestyle": style["linestyle"],
            "label": style["label"],
        }
        axes[0].plot(
            report["sequence_length_bp"],
            report["mean_seconds"] * 1_000,
            **shared,
        )
        axes[1].plot(
            report["sequence_length_bp"],
            report["peak_allocated_gib"],
            **shared,
        )

    lengths = sorted(
        {length for report in reports.values() for length in report["sequence_length_bp"]}
    )
    axes[0].set_title("Mean Latency")
    axes[0].set_ylabel("Latency (ms/batch)")
    axes[1].set_title("Peak GPU Memory")
    axes[1].set_ylabel("Allocated Memory (GiB)")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(lengths, [format_length(length) for length in lengths])
        axis.set_xlabel("Sequence Length (bp)")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(f"{benchmark.title()} Performance", fontsize=14, y=0.97)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(labels),
        frameon=False,
    )
    output = RESULTS_DIR / f"{benchmark}-performance.svg"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


def main() -> None:
    for benchmark in ("inference", "training"):
        plot_benchmark(benchmark)


if __name__ == "__main__":
    main()
