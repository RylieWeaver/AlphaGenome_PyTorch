# External
from collections.abc import Sequence
import random
from typing import Optional, Union

import torch

# Internal
from alphagenome_pt import (
    AlphaGenome,
    DataBatch,
    Metadata,
    small_alphagenome,
    synthetic_batch,
    synthetic_metadata,
)


    
def resolve_device(device: Optional[Union[torch.device, str]] = None) -> torch.device:
    # NOTE: Only allow CPU training if not doing any type of parallelism
    if isinstance(device, torch.device):
        return device
    elif isinstance(device, str):
        return torch.device(device)
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def move_to(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to(v, device) for v in data]
    else:
        return data
    
def bert_mlm(
    sequences: list[str],
    mask_token: str = 'N',
    select_prob: float = 0.15,
    mask_prob: float = 0.8,
    random_prob = 0.1,
    keep_prob = 0.1
) -> tuple[list[str], list[str]]:
    """Applies BERT-style masked language modeling to the input DNA sequences."""
    masked_sequences = []
    labels = []
    for seq in sequences:
        masked_seq = list(seq)
        label_seq = [mask_token] * len(seq)
        for i in range(len(seq)):
            if random.random() < select_prob:
                rand_val = random.random()
                if rand_val < mask_prob:
                    # Mask the token
                    masked_seq[i] = mask_token
                elif rand_val < mask_prob + random_prob:
                    # Replace with a random token
                    masked_seq[i] = random.choice('ACGT')
                else:
                    # Keep the original token (but still predict it)
                    pass
                label_seq[i] = seq[i]  # Set the label to the original token
        masked_sequences.append("".join(masked_seq))
        labels.append("".join(label_seq))
    return masked_sequences, labels


def synthetic_data_splits(
    metadata: Metadata,
    *,
    num_train_batches: int = 40,
    num_val_batches: int = 5,
    num_test_batches: int = 5,
    batch_size: int = 2,
    seq_len: int = 8192,
    num_splice_sites: int | None = None,
) -> dict[str, list[DataBatch]]:
    def batches(num_batches: int) -> list[DataBatch]:
        return [
            synthetic_batch(
                metadata,
                batch_size=batch_size,
                seq_len=seq_len,
                num_splice_sites=num_splice_sites,
            )
            for _ in range(num_batches)
        ]

    return {
        "train": batches(num_train_batches),
        "val": batches(num_val_batches),
        "test": batches(num_test_batches),
    }


def _model_device(model: AlphaGenome) -> torch.device:
    return next(model.parameters()).device


def example_eval(
    model: AlphaGenome,
    batches: Sequence[DataBatch],
    *,
    name: str,
) -> None:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in batches:
            loss, _, _ = model.loss(batch.to(_model_device(model)))
            losses.append(float(loss.detach().cpu()))
    print(f"{name}: loss={sum(losses) / len(losses):.6g}")


def example_train(
    model: AlphaGenome,
    batches: Sequence[DataBatch],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    start_step: int = 0,
) -> None:
    model.train()
    for local_step in range(steps):
        step = start_step + local_step
        batch = batches[step % len(batches)].to(_model_device(model))
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model.loss(batch)
        loss.backward()
        optimizer.step()
        print(f"train step {step + 1}: loss={float(loss.detach().cpu()):.6g}")


def example_train_val_test(
    metadata: Metadata | None = None,
    model: AlphaGenome | None = None,
    *,
    steps_per_eval: int = 20,
    max_steps: int = 100,
    num_train_batches: int = 40,
    num_val_batches: int = 5,
    num_test_batches: int = 5,
    batch_size: int = 2,
    seq_len: int | None = None,
    lr: float = 1e-4,
    optimizer: torch.optim.Optimizer | None = None,
    **cfg_overrides,
) -> dict[str, object]:
    if metadata is None:
        metadata = model.metadata if model is not None else synthetic_metadata()
    if model is None:
        model = small_alphagenome(metadata, **cfg_overrides)
    if seq_len is None:
        seq_len = model.max_seq_len
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    data = synthetic_data_splits(
        metadata,
        num_train_batches=num_train_batches,
        num_val_batches=num_val_batches,
        num_test_batches=num_test_batches,
        batch_size=batch_size,
        seq_len=seq_len,
        num_splice_sites=model.num_splice_sites,
    )

    print("eval step 0")
    example_eval(model, data["val"], name="val")
    example_eval(model, data["test"], name="test")

    completed_steps = 0
    while completed_steps < max_steps:
        steps = min(steps_per_eval, max_steps - completed_steps)
        example_train(
            model,
            data["train"],
            optimizer,
            steps=steps,
            start_step=completed_steps,
        )
        completed_steps += steps
        print(f"eval step {completed_steps}")
        example_eval(model, data["val"], name="val")
        example_eval(model, data["test"], name="test")

    return {
        "model": model,
        "optimizer": optimizer,
        "data": data,
    }
