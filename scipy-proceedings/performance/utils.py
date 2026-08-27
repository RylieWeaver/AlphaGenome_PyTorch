"""Shared utilities for the inference and training performance benchmarks."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from alphagenome_pt import deepmind_metadata, package_version


METRIC_FIELDNAMES = (
    "sequence_length_bp",
    "batch_size",
    "warmup_iterations",
    "timed_iterations",
    "mean_seconds",
    "median_seconds",
    "sequences_per_second",
    "bp_per_second",
    "baseline_allocated_bytes",
    "peak_allocated_bytes",
    "peak_working_bytes",
    "baseline_allocated_gib",
    "peak_allocated_gib",
    "peak_working_gib",
)
RUN_FIELDNAMES = ("seed", "status", "error")
ENVIRONMENT_FIELDNAMES = (
    "gpu",
    "gpu_total_memory_bytes",
    "pytorch_version",
    "cuda_version",
    "alphagenome_pt_version",
)
BENCHMARK_MODES = ("predictions", "embeddings")


def metadata_for_mode(mode: str) -> dict:
    metadata = deepmind_metadata()
    if mode == "embeddings":
        return {**metadata, "heads": {}}
    return metadata


def validate_args(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.warmup <= 0 or args.repetitions <= 0:
        raise ValueError("--warmup and --repetitions must be positive.")
    invalid = [
        length
        for length in args.sequence_lengths
        if length < 2_048 or length > 1_048_576 or length % 2_048
    ]
    if invalid:
        raise ValueError(
            "Sequence lengths must be within [2,048, 1,048,576] and divisible "
            f"by 2,048; invalid values: {invalid}"
        )


def summarize_measurements(
    elapsed_ms: Sequence[float],
    *,
    sequence_length: int,
    batch_size: int,
    warmup: int,
    repetitions: int,
    baseline: int,
    peak: int,
) -> dict[str, object]:
    mean_seconds = statistics.fmean(elapsed_ms) / 1_000
    median_seconds = statistics.median(elapsed_ms) / 1_000
    sequences_per_second = batch_size / mean_seconds
    return {
        "sequence_length_bp": sequence_length,
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "timed_iterations": repetitions,
        "mean_seconds": f"{mean_seconds:.9f}",
        "median_seconds": f"{median_seconds:.9f}",
        "sequences_per_second": f"{sequences_per_second:.9f}",
        "bp_per_second": f"{sequences_per_second * sequence_length:.3f}",
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_working_bytes": peak - baseline,
        "baseline_allocated_gib": f"{baseline / 2**30:.6f}",
        "peak_allocated_gib": f"{peak / 2**30:.6f}",
        "peak_working_gib": f"{(peak - baseline) / 2**30:.6f}",
        "status": "ok",
        "error": "",
    }


def failure_row(sequence_length: int, args, error: Exception) -> dict[str, object]:
    return {
        "sequence_length_bp": sequence_length,
        "batch_size": args.batch_size,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.repetitions,
        "status": "out_of_memory",
        "error": " ".join(str(error).splitlines()),
    }


def environment_metadata(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "alphagenome_pt_version": package_version(),
    }


def write_report(
    output: Path,
    rows: list[dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=fieldnames).to_csv(output, index=False)
