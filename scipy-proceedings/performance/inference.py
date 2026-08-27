#!/usr/bin/env python3
"""Benchmark AlphaGenome inference throughput and CUDA memory."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from alphagenome_pt import (
    DNAOneHotEncoder,
    DataBatch,
    deepmind_model,
)
from utils import (
    BENCHMARK_MODES,
    ENVIRONMENT_FIELDNAMES,
    METRIC_FIELDNAMES,
    RUN_FIELDNAMES,
    environment_metadata,
    failure_row,
    metadata_for_mode,
    summarize_measurements,
    validate_args,
    write_report,
)


DEFAULT_SEQUENCE_LENGTHS = (
    2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144
)
FIELDNAMES = METRIC_FIELDNAMES + (
    "mode",
    "heads",
    "dtype_policy",
) + RUN_FIELDNAMES + ENVIRONMENT_FIELDNAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTHS,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype-policy", default="deepmind")
    parser.add_argument("--mode", choices=BENCHMARK_MODES, default="predictions")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def make_batch(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> DataBatch:
    sequences = [
        "".join(random.choices("ACGT", k=sequence_length))
        for _ in range(batch_size)
    ]
    one_hot = DNAOneHotEncoder().encode(sequences)
    return DataBatch(
        dna_sequence_one_hot=one_hot.to(device),
        organism_index=torch.zeros(batch_size, dtype=torch.long, device=device),
    )


def benchmark(
    model,
    sequence_length: int,
    *,
    batch_size: int,
    warmup: int,
    repetitions: int,
    device: torch.device,
    mode: str,
) -> dict[str, object]:
    batch = make_batch(batch_size, sequence_length, device)
    model_mode = "predict" if mode == "predictions" else "embed"

    with torch.inference_mode():
        for _ in range(warmup):
            outputs = model(batch, mode=model_mode)
            del outputs
        
        torch.cuda.synchronize(device)

        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed_ms: list[float] = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = model(batch, mode=model_mode)
            end.record()
            end.synchronize()
            elapsed_ms.append(start.elapsed_time(end))
            del outputs

        peak = torch.cuda.max_memory_allocated(device)

    return summarize_measurements(
        elapsed_ms,
        sequence_length=sequence_length,
        batch_size=batch_size,
        warmup=warmup,
        repetitions=repetitions,
        baseline=baseline,
        peak=peak,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model = deepmind_model(
        device=device,
        metadata=metadata_for_mode(args.mode),
        load_state=False,  # only need arch for throughput benchmarking
        dtype_policy=args.dtype_policy,
    )
    model.eval()

    common = {
        "mode": args.mode,
        "heads": ",".join(model._heads.keys()),
        "dtype_policy": args.dtype_policy,
        "seed": args.seed,
        **environment_metadata(device),
    }
    rows: list[dict[str, object]] = []
    for sequence_length in args.sequence_lengths:
        try:
            row = benchmark(
                model,
                sequence_length,
                batch_size=args.batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
                device=device,
                mode=args.mode,
            )
        except torch.cuda.OutOfMemoryError as error:
            row = failure_row(sequence_length, args, error)
            torch.cuda.empty_cache()
        rows.append({**row, **common})

    output = args.output or (
        Path(__file__).resolve().parents[1]
        / "results"
        / f"inference-{args.mode}.csv"
    )
    write_report(output, rows, FIELDNAMES)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
