#!/usr/bin/env python3
"""Plot training curves from a proceedings result directory."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOSS_COLUMNS = {
    "train_with_grad_loss": "Train (optimization batch)",
    "train_without_grad_loss": "Train (evaluation)",
    "validation_loss": "Validation",
    "test_loss": "Test",
}
ACCURACY_COLUMNS = {
    "train_masked_accuracy": "Train",
    "validation_masked_accuracy": "Validation",
    "test_masked_accuracy": "Test",
}
COLORS = {
    "Train (optimization batch)": "#7f7f7f",
    "Train (evaluation)": "#1f77b4",
    "Train": "#1f77b4",
    "Validation": "#ff7f0e",
    "Test": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def plot_lines(ax, metrics, columns) -> None:
    for column, label in columns.items():
        if column in metrics:
            values = metrics[column].notna()
            ax.plot(
                metrics.loc[values, "step"],
                metrics.loc[values, column],
                marker="o",
                markersize=4,
                linewidth=2,
                color=COLORS[label],
                label=label,
            )


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.result_dir / "metrics.csv")
    with (args.result_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)

    has_accuracy = "validation_masked_accuracy" in metrics
    figure, axes = plt.subplots(
        1,
        2 if has_accuracy else 1,
        figsize=(11 if has_accuracy else 6.5, 4.5),
        squeeze=False,
    )
    figure.subplots_adjust(top=0.84, wspace=0.28)
    loss_axis = axes[0, 0]
    plot_lines(loss_axis, metrics, LOSS_COLUMNS)
    loss_axis.set_title("Loss")
    loss_axis.set_ylabel("Loss")

    if has_accuracy:
        accuracy_axis = axes[0, 1]
        plot_lines(accuracy_axis, metrics, ACCURACY_COLUMNS)
        accuracy_axis.set_title("Accuracy")
        accuracy_axis.set_ylabel("Accuracy")
        accuracy_axis.set_ylim(0, 1)

    for axis in axes.flat:
        axis.set_xlabel("Training Step")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)

    task = metadata["task"].replace("_", " ").title()
    initialization = metadata["initialization"].title()
    figure.suptitle(
        f"{task} — {initialization} Initialization",
        fontsize=14,
        y=0.97,
    )
    output = args.result_dir / "training-curves.svg"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
