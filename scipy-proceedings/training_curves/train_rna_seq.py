#!/usr/bin/env python3
"""Train the published AlphaGenome architecture with an RNA-seq head from indexed FASTA/BigWig windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from alphagenome_pt import Metadata, deepmind_model
from utils import (
    DEFAULT_BIGWIG,
    DEFAULT_FASTA,
    output_dir,
    save_results,
    validate_args,
)
from data import (
    RnaSeqWindowDataset,
    calculate_nonzero_mean,
    get_loaders,
    rna_seq_batch,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--bigwig", type=Path, default=DEFAULT_BIGWIG)
    parser.add_argument("--chromosome", default="chr1")
    parser.add_argument("--sequence-length", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--dtype-policy", default="deepmind")
    parser.add_argument(
        "--load-state",
        action="store_true",
        help="Initialize the published backbone from its checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def evaluate(model, loaders, *, device: torch.device) -> dict[str, float]:
    model.eval()
    values = {}
    with torch.inference_mode():
        for split, loader in loaders.items():
            losses = [
                model(
                    rna_seq_batch(dna_sequence_one_hot, rna_signal, device=device),
                    mode="loss",
                ).total.item()
                for dna_sequence_one_hot, rna_signal in loader
            ]
            values[
                "train_without_grad_loss" if split == "train" else f"{split}_loss"
            ] = sum(losses) / len(losses)
    return values


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    initialization = "checkpoint" if args.load_state else "from-scratch"

    with RnaSeqWindowDataset(
        fasta_path=args.fasta,
        bigwig_path=args.bigwig,
        chromosome=args.chromosome,
        window_size=args.sequence_length,
    ) as dataset:
        train_dataset, val_dataset, test_dataset = split_dataset(dataset)
        train_loader, evaluation_loaders = get_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            seed=args.seed,
        )

        track_mean = calculate_nonzero_mean(dataset, train_dataset.indices)
        print(f"Training-split nonzero RNA-seq mean: {track_mean:.6f}")

        metadata = Metadata(
            {
                "organisms": ["human"],
                "heads": {
                    "rna_seq": {
                        "num_tracks": [1],
                        "means": [[track_mean]],
                    }
                },
            }
        )
        model = deepmind_model(
            device=device,
            metadata=metadata,
            dtype_policy=args.dtype_policy,
            load_state=args.load_state,
            heads=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

        rows: list[dict[str, object]] = []
        evaluation = evaluate(
            model,
            evaluation_loaders,
            device=device,
        )
        rows.append(
            {
                "step": 0,
                "train_with_grad_loss": "",
                **evaluation,
            }
        )
        print(
            "step=0 train_with_grad=N/A "
            f"train={evaluation['train_without_grad_loss']:.6f} "
            f"validation={evaluation['validation_loss']:.6f} "
            f"test={evaluation['test_loss']:.6f}"
        )

        step = 0
        train_with_grad_losses: list[float] = []
        while step < args.steps:
            for dna_sequence_one_hot, rna_signal in train_loader:
                step += 1
                model.train()
                batch = rna_seq_batch(dna_sequence_one_hot, rna_signal, device=device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch, mode="loss")
                output.total.backward()
                optimizer.step()
                train_with_grad_losses.append(output.total.detach().item())
                del output, batch

                if step % args.eval_every == 0 or step == args.steps:
                    train_with_grad_loss = (
                        sum(train_with_grad_losses) / len(train_with_grad_losses)
                    )
                    evaluation = evaluate(
                        model,
                        evaluation_loaders,
                        device=device,
                    )
                    rows.append(
                        {
                            "step": step,
                            "train_with_grad_loss": train_with_grad_loss,
                            **evaluation,
                        }
                    )
                    print(
                        f"step={step} train_with_grad={train_with_grad_loss:.6f} "
                        f"train={evaluation['train_without_grad_loss']:.6f} "
                        f"validation={evaluation['validation_loss']:.6f} "
                        f"test={evaluation['test_loss']:.6f}"
                    )
                    train_with_grad_losses.clear()

                if step == args.steps:
                    break

    result_dir = output_dir(args, "rna-seq")
    save_results(
        result_dir,
        rows,
        {
            "task": "rna_seq",
            "architecture": "published",
            "initialization": initialization,
            "dtype_policy": args.dtype_policy,
            "seed": args.seed,
            "fasta": str(args.fasta),
            "bigwig": str(args.bigwig),
            "chromosome": args.chromosome,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "eval_batches": args.eval_batches,
            "learning_rate": 3e-5,
            "track_nonzero_mean": track_mean,
        },
    )
    print(f"Wrote {len(rows)} points to {result_dir}")


if __name__ == "__main__":
    main()
