"""Plot precision trends from a JAX-PyTorch equivalence report.

This script generates the figures used in ``docs/deepmind-equivalence.md``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator


POLICIES = ("deepmind", "float32", "float64")
POLICY_LABELS = ("DeepMind\nBF16", "FP32", "FP64")
METRICS = ("relative_L2", "relative_Linf")
METRIC_LABELS = {
    "relative_L2": r"Relative $L_2$ error",
    "relative_Linf": r"Relative $L_\infty$ error",
}
GENOME_HEADS = {
    "atac",
    "cage",
    "chip_histone",
    "chip_tf",
    "dnase",
    "procap",
    "rna_seq",
}
HEAD_LABELS = {
    "atac": "ATAC",
    "cage": "CAGE",
    "chip_histone": "ChIP histone",
    "chip_tf": "ChIP TF",
    "contact_maps": "Contact maps",
    "dnase": "DNase",
    "procap": "ProCap",
    "rna_seq": "RNA-seq",
    "splice_sites_classification": "Splice classification",
    "splice_sites_junction": "Splice junction",
    "splice_sites_usage": "Splice usage",
}
COLORS = (
    "#2563EB",
    "#D97706",
    "#DB2777",
    "#4D7C0F",
    "#7C3AED",
)
LEGEND_COLOR = "#64748B"
RESOLUTION_STYLES = {
    "1bp": ("-", "o"),
    "128bp": ("--", "s"),
    "pair": (":", "D"),
    "other": ("-.", "^"),
}

FIGURE_TITLE_Y = 0.97
FIGURE_HEADER_STEP = 0.08
FIGURE_SECOND_ROW_Y = FIGURE_TITLE_Y - FIGURE_HEADER_STEP
FIGURE_CONTENT_GAP = 0.03
FULL_MODEL_CONTENT_TOP = FIGURE_TITLE_Y - FIGURE_CONTENT_GAP
ENCODER_LEGEND_Y = FIGURE_SECOND_ROW_Y
ENCODER_CONTENT_TOP = ENCODER_LEGEND_Y - FIGURE_CONTENT_GAP
ENCODER_TICK_COUNT = 4
ENCODER_AXIS_PADDING_DECADES = 0.75
ENCODER_LOWER_DECADE_OVERRIDES = {"float64": -17}

def _series_label(representation: str) -> str:
    parts = representation.split("/")
    if parts[0] == "embeddings":
        return {"1bp": "1 bp", "128bp": "128 bp", "pair": "Pair"}[parts[1]]
    if parts[0] == "loss":
        return HEAD_LABELS.get(parts[1], parts[1].replace("_", " ").title())
    head = HEAD_LABELS.get(parts[1], parts[1].replace("_", " ").title())
    output = parts[2]
    if output.endswith("_1bp"):
        return f"{head} · 1 bp"
    if output.endswith("_128bp"):
        return f"{head} · 128 bp"
    return head


def _plot_precision_panel(
    ax,
    rows: pd.DataFrame,
    metric: str,
    title: str,
):
    values = rows.pivot(index="representation", columns="dtype_policy", values=metric)
    if values.columns.tolist() != list(POLICIES):
        values = values.reindex(columns=POLICIES)
    if values.isna().any().any():
        raise ValueError(f"{title} does not have one value for every precision policy")
    x = np.arange(len(POLICIES))
    tasks = sorted(
        {
            name.split("/")[0]
            if name.startswith("embeddings/")
            else name.split("/")[1]
            for name in values.index
        }
    )
    task_colors = {
        task: COLORS[index % len(COLORS)]
        for index, task in enumerate(tasks)
    }
    for name, row in values.iterrows():
        parts = name.split("/")
        task = parts[0] if parts[0] == "embeddings" else parts[1]
        output = parts[-1]
        resolution = next(
            (key for key in ("1bp", "128bp", "pair") if output.endswith(key)),
            "other",
        )
        linestyle, marker = RESOLUTION_STYLES[resolution]
        # Exact matches have zero error, which cannot be represented on a log
        # axis. Omit them while still connecting the positive measurements for
        # the remaining precision policies.
        row = row.to_numpy()
        positive = row > 0
        ax.plot(
            x[positive],
            row[positive],
            color=task_colors[task],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=_series_label(name),
        )
    x_limits = (-0.12, len(POLICIES) - 0.88)
    ax.set_title(title, loc="left", fontsize=12, color="#0F172A", pad=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(POLICIES)), POLICY_LABELS)
    ax.set_xlim(x_limits)
    ax.set_xlabel("Precision policy (discrete)", color="#334155", fontsize=10)
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94A3B8")
    ax.tick_params(colors="#475569", labelsize=9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="best",
        fontsize=6.8,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        ncol=2 if len(labels) > 7 else 1,
        columnspacing=0.8,
        handlelength=2.4,
    )


def _select_full_model_panels(report: pd.DataFrame):
    full = report[
        report["test_name"].str.contains("test_full_model[", regex=False)
    ].copy()
    representations = full["representation"]

    embeddings = full[representations.str.startswith("embeddings/")]
    descaled = full[
        representations.str.match(
            rf"heads/({'|'.join(sorted(GENOME_HEADS))})/predictions_(1|128)bp$"
        )
    ]
    scaled_genome = representations.str.match(
        rf"heads/({'|'.join(sorted(GENOME_HEADS))})/scaled_predictions_(1|128)bp$"
    )
    direct_predictions = representations.isin(
        (
            "heads/contact_maps/predictions",
            "heads/splice_sites_classification/predictions",
            "heads/splice_sites_junction/predictions",
            "heads/splice_sites_usage/predictions",
        )
    )
    model_space = full[scaled_genome | direct_predictions]
    losses = full[
        representations.str.match(r"loss/[^/]+$")
        & representations.ne("loss/total")
    ]
    return embeddings, model_space, descaled, losses


def _plot_missing_losses(ax):
    ax.set_title("Per-head losses", loc="left", fontsize=12, color="#0F172A")
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "Not recorded in this report",
        ha="center",
        va="center",
        fontsize=13,
        color="#334155",
        weight="bold",
    )
    ax.text(
        0.5,
        0.44,
        "Full-model losses require a sequence length of at least 131,072 bp.",
        ha="center",
        va="center",
        fontsize=9,
        color="#64748B",
        wrap=True,
    )


def plot_full_model(report, metric, output_dir):
    panels = _select_full_model_panels(report)
    titles = (
        "Embeddings",
        "Model-space head predictions",
        "Descaled genome-track predictions",
        "Per-head losses",
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10.5),
        sharey=True,
        constrained_layout=False,
    )
    for ax, rows, title in zip(axes.flat, panels, titles):
        if rows.empty and title == "Per-head losses":
            _plot_missing_losses(ax)
        elif rows.empty:
            raise ValueError(f"No rows available for full-model panel {title!r}")
        else:
            _plot_precision_panel(ax, rows, metric, title)
    fig.suptitle(
        f"{METRIC_LABELS[metric].removesuffix(' error')} Error for Full Model Outputs",
        fontsize=17,
        color="#0F172A",
        y=FIGURE_TITLE_Y,
    )
    fig.tight_layout(rect=(0, 0, 1, FULL_MODEL_CONTENT_TOP), h_pad=2.0)
    _save_figure(fig, output_dir / f"full-model-{metric.lower()}.svg")


def plot_encoder(report, metric, output_dir):
    rows = report[
        report["test_name"].str.contains("test_encoder_by_layer", regex=False)
    ].copy()
    rows["bin_size"] = (
        rows["representation"].str.extract(r"bin_size_(\d+)").astype(int)
    )
    rows = rows.sort_values(["dtype_policy", "bin_size"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0))
    for ax, policy, label, color in zip(axes, POLICIES, POLICY_LABELS, COLORS):
        selected = rows[rows["dtype_policy"].eq(policy)]
        if selected.empty:
            raise ValueError(f"No encoder rows available for {policy!r}")
        values = selected[metric].to_numpy(dtype=float)
        stage_indices = np.arange(len(selected))
        if np.any(values <= 0):
            raise ValueError(
                f"Encoder {metric} values must be positive for a log-scale plot"
            )
        median_exponent = np.median(np.log10(values))
        centered_lower_decade = int(
            np.rint(median_exponent - (ENCODER_TICK_COUNT - 1) / 2)
        )
        lower_decade = ENCODER_LOWER_DECADE_OVERRIDES.get(
            policy, centered_lower_decade
        )
        upper_decade = lower_decade + ENCODER_TICK_COUNT - 1
        ticks = 10.0 ** np.arange(lower_decade, upper_decade + 1)
        lower_limit = 10.0 ** (
            lower_decade - ENCODER_AXIS_PADDING_DECADES
        )
        upper_limit = 10.0 ** (
            upper_decade + ENCODER_AXIS_PADDING_DECADES
        )
        if np.any((values < lower_limit) | (values > upper_limit)):
            raise ValueError(
                f"Encoder {metric} values do not fit in the shared log template"
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_ylim(lower_limit, upper_limit)
        ax.plot(
            stage_indices,
            values,
            color=color,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.4,
            linewidth=2,
            label="Observed error",
        )
        ax.set_xticks(stage_indices)
        ax.set_xlabel("Encoder stage", color="#334155")
        ax.set_title(label.replace("\n", " "), fontsize=12, color="#0F172A")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#94A3B8")
        ax.tick_params(colors="#475569", labelsize=9)
    legend_handle = Line2D(
        [],
        [],
        color=LEGEND_COLOR,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.4,
        linewidth=2,
        label="Observed error",
    )
    fig.suptitle(
        f"{METRIC_LABELS[metric].removesuffix(' error')} Error Through Encoder Stages",
        fontsize=17,
        color="#0F172A",
        y=FIGURE_TITLE_Y,
    )
    fig.legend(
        [legend_handle],
        [legend_handle.get_label()],
        loc="upper center",
        bbox_to_anchor=(0.5, ENCODER_LEGEND_Y),
        ncol=1,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, ENCODER_CONTENT_TOP))
    _save_figure(fig, output_dir / f"encoder-{metric.lower()}.svg")



def _save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/_static/deepmind-equivalence"),
    )
    args = parser.parse_args()
    report = pd.read_csv(args.report)
    required = {"test_name", "dtype_policy", "representation", *METRICS}
    if missing := required - set(report.columns):
        raise ValueError(f"Missing report columns: {sorted(missing)}")

    panels = _select_full_model_panels(report)
    titles = (
        "Embeddings",
        "Model-space predictions",
        "Descaled predictions",
        "Per-head losses",
    )
    print(f"Reading {len(report)} rows from {args.report}")
    for title, rows in zip(titles, panels):
        names = ", ".join(sorted(rows.representation.unique())) or "none recorded"
        print(f"  {title} ({rows.representation.nunique()} series): {names}")

    for metric in METRICS:
        plot_full_model(report, metric, args.output_dir)
        plot_encoder(report, metric, args.output_dir)
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
