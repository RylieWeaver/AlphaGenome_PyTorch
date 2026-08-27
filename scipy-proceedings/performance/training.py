#!/usr/bin/env python3
"""Benchmark AlphaGenome training throughput and CUDA memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from alphagenome_pt import deepmind_model, synthetic_batch
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
    2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072
)
FIELDNAMES = METRIC_FIELDNAMES + (
    "mode",
    "heads",
    "optimizer",
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


def embedding_loss(embs_tuple, targets) -> torch.Tensor:
    return sum(
        F.mse_loss(value, target)
        for value, target in zip(
            embs_tuple, targets, strict=True
        )
    )


def train_step(model, optimizer, batch, model_mode: str, targets) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch, mode=model_mode)
    if model_mode == "loss":
        loss = outputs.total
    else:
        embs_tuple = (
            outputs.embeddings_1bp,
            outputs.embeddings_128bp,
            outputs.embeddings_pair,
        )
        loss = embedding_loss(embs_tuple, targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def benchmark(
    model,
    optimizer,
    sequence_length: int,
    *,
    batch_size: int,
    warmup: int,
    repetitions: int,
    device: torch.device,
    mode: str,
) -> dict[str, object]:
    batch = synthetic_batch(
        model.metadata,
        batch_size=batch_size,
        seq_len=sequence_length,
        num_splice_sites=model.num_splice_sites,
    ).to(device)
    model_mode = "loss" if mode == "predictions" else "embed"
    
    targets = None
    if model_mode == "embed":
        with torch.no_grad():
            embeddings = model(batch, mode=model_mode)
        embs_tuple = (
            embeddings.embeddings_1bp,
            embeddings.embeddings_128bp,
            embeddings.embeddings_pair,
        )
        targets = tuple(torch.randn_like(value) for value in embs_tuple)
        del embeddings, embs_tuple

    for _ in range(warmup):
        loss = train_step(model, optimizer, batch, model_mode, targets)
        del loss

    torch.cuda.synchronize(device)

    optimizer.zero_grad(set_to_none=True)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = train_step(model, optimizer, batch, model_mode, targets)
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))
        del loss

    peak = torch.cuda.max_memory_allocated(device)
    optimizer.zero_grad(set_to_none=True)
    del batch

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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model = deepmind_model(
        device=device,
        metadata=metadata_for_mode(args.mode),
        load_state=False,  # only need arch for throughput benchmarking
        dtype_policy=args.dtype_policy,
    )
    model.train()

    lr = 3e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    common = {
        "mode": args.mode,
        "heads": ",".join(model._heads.keys()),
        "optimizer": f"AdamW(lr={lr})",
        "dtype_policy": args.dtype_policy,
        "seed": args.seed,
        **environment_metadata(device),
    }

    rows: list[dict[str, object]] = []
    for sequence_length in args.sequence_lengths:
        try:
            row = benchmark(
                model,
                optimizer,
                sequence_length,
                batch_size=args.batch_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
                device=device,
                mode=args.mode,
            )
        except torch.cuda.OutOfMemoryError as error:
            row = failure_row(sequence_length, args, error)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        rows.append({**row, **common})

    output = args.output or (
        Path(__file__).resolve().parents[1]
        / "results"
        / f"training-{args.mode}.csv"
    )
    write_report(output, rows, FIELDNAMES)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
