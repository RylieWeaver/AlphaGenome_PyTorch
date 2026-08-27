#!/usr/bin/env python3
"""Train the published AlphaGenome architecture for MLM on indexed windows from one chromosome."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from alphagenome_pt import Metadata, deepmind_model
from utils import DEFAULT_FASTA, output_dir, save_results, validate_args
from data import (
    FastaWindowDataset,
    get_loaders,
    mask_mlm_batch,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
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


def evaluate(model, loaders, *, device: torch.device, seed: int) -> dict[str, float]:
    model.eval()
    values = {}
    with torch.inference_mode():
        for split, loader in loaders.items():
            mask_generator = torch.Generator().manual_seed(seed)
            losses: list[float] = []
            correct = 0
            total = 0
            for dna_sequence_one_hot in loader:
                batch = mask_mlm_batch(
                    dna_sequence_one_hot,
                    device=device,
                    generator=mask_generator,
                )
                output = model(batch, mode="loss", return_predictions=True)
                losses.append(output.total.item())
                labels = batch.mlm
                assert labels is not None and output.predictions is not None
                logits = output.predictions["masked_language_modeling"]["logits"]
                masked = labels != -100
                correct += (
                    logits.argmax(dim=-1)[masked] == labels[masked]
                ).sum().item()
                total += masked.sum().item()
            values[
                "train_without_grad_loss" if split == "train" else f"{split}_loss"
            ] = sum(losses) / len(losses)
            values[f"{split}_masked_accuracy"] = correct / total
    return values


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    train_mask_generator = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda")
    initialization = "checkpoint" if args.load_state else "from-scratch"

    with FastaWindowDataset(
        fasta_path=args.fasta,
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

        metadata = Metadata(
            {
                "organisms": ["human"],
                "heads": {"masked_language_modeling": {}},
            }
        )
        model = deepmind_model(
            device=device,
            metadata=metadata,
            dtype_policy=args.dtype_policy,
            load_state=args.load_state,
            heads=False,
        )
        lr = 3e-5
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        rows: list[dict[str, object]] = []
        evaluation = evaluate(
            model,
            evaluation_loaders,
            device=device,
            seed=args.seed,
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
            f"test={evaluation['test_loss']:.6f} "
            f"train_accuracy={evaluation['train_masked_accuracy']:.6f} "
            f"validation_accuracy={evaluation['validation_masked_accuracy']:.6f} "
            f"test_accuracy={evaluation['test_masked_accuracy']:.6f}"
        )

        step = 0
        train_with_grad_losses: list[float] = []
        while step < args.steps:
            for dna_sequence_one_hot in train_loader:
                step += 1
                model.train()
                batch = mask_mlm_batch(
                    dna_sequence_one_hot,
                    device=device,
                    generator=train_mask_generator,
                )
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
                        seed=args.seed,
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
                        f"test={evaluation['test_loss']:.6f} "
                        f"train_accuracy={evaluation['train_masked_accuracy']:.6f} "
                        f"validation_accuracy={evaluation['validation_masked_accuracy']:.6f} "
                        f"test_accuracy={evaluation['test_masked_accuracy']:.6f}"
                    )
                    train_with_grad_losses.clear()

                if step == args.steps:
                    break

    result_dir = output_dir(args, "mlm")
    save_results(
        result_dir,
        rows,
        {
            "task": "masked_language_modeling",
            "architecture": "published",
            "initialization": initialization,
            "dtype_policy": args.dtype_policy,
            "seed": args.seed,
            "fasta": str(args.fasta),
            "chromosome": args.chromosome,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "eval_batches": args.eval_batches,
            "learning_rate": lr,
        },
    )
    print(f"Wrote {len(rows)} points to {result_dir}")


if __name__ == "__main__":
    main()
