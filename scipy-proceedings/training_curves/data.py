"""Shared chromosome-window sampling for the proceedings training curves."""

from __future__ import annotations

import numpy as np
import pyBigWig
import pysam
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from alphagenome_pt import DNAOneHotEncoder, DataBatch


class FastaWindowDataset(Dataset):
    """Fixed one-hot DNA windows with 25% overlap from one chromosome."""

    def __init__(
        self,
        *,
        fasta_path,
        chromosome: str,
        window_size: int,
    ):
        self.fasta = pysam.FastaFile(str(fasta_path))
        chromosome_size = chromosome_length(self.fasta, chromosome)
        if chromosome_size < window_size:
            raise ValueError("Chromosome is shorter than one window.")

        self.chromosome = chromosome
        self.chromosome_size = chromosome_size
        self.window_size = window_size
        self.encoder = DNAOneHotEncoder()
        stride = 3 * window_size // 4
        self.starts = tuple(
            range(0, chromosome_size - window_size + 1, stride)
        )

    def close(self) -> None:
        self.fasta.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> torch.Tensor:
        start = self.starts[index]
        sequence = self.fasta.fetch(
            self.chromosome, start, start + self.window_size
        )
        if len(sequence) != self.window_size:
            raise RuntimeError(
                f"Expected {self.window_size} bases, received {len(sequence)}."
            )
        return self.encoder.encode(sequence)


class RnaSeqWindowDataset(FastaWindowDataset):
    """Aligned DNA and one-channel RNA-seq windows."""

    def __init__(
        self,
        *,
        fasta_path,
        bigwig_path,
        chromosome: str,
        window_size: int,
    ):
        self.bigwig = None
        try:
            super().__init__(
                fasta_path=fasta_path,
                chromosome=chromosome,
                window_size=window_size,
            )
            self.bigwig = pyBigWig.open(str(bigwig_path))
            chromosome_length(self.fasta, chromosome, self.bigwig)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.bigwig is not None:
            self.bigwig.close()
            self.bigwig = None
        super().close()

    def get_rna_signal(self, start: int) -> torch.Tensor:
        rna_signal = read_rna_signal(
            bigwig=self.bigwig,
            chromosome=self.chromosome,
            start=start,
            window_size=self.window_size,
        )
        return rna_signal[:, None]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dna_sequence_one_hot = super().__getitem__(index)
        return dna_sequence_one_hot, self.get_rna_signal(self.starts[index])


def split_dataset(
    dataset: FastaWindowDataset,
) -> tuple[Subset, Subset, Subset]:
    """Split genomic regions 80/10/10 without overlapping across splits."""

    train_end = int(dataset.chromosome_size * 0.8)
    validation_end = int(dataset.chromosome_size * 0.9)
    regions = (
        (0, train_end),
        (train_end, validation_end),
        (validation_end, dataset.chromosome_size),
    )
    split_indices = tuple(
        [
            index
            for index, start in enumerate(dataset.starts)
            if start >= region_start
            and start + dataset.window_size <= region_end
        ]
        for region_start, region_end in regions
    )
    if any(not indices for indices in split_indices):
        raise ValueError("Dataset is too small for an 80/10/10 split.")
    return tuple(Subset(dataset, indices) for indices in split_indices)


def get_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    *,
    batch_size: int,
    eval_batches: int,
    seed: int,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Build the training loader and fixed evaluation loaders."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    evaluation_loaders = {}
    for split, dataset in {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }.items():
        num_samples = min(eval_batches * batch_size, len(dataset))
        indices = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(seed),
        )[:num_samples].tolist()
        evaluation_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            # Keep a consistent, randomly chosen evaluation subset.
            sampler=indices,
        )
    return train_loader, evaluation_loaders


def mask_mlm_batch(
    dna_sequence_one_hot: torch.Tensor,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataBatch:
    """Mask 15% of one-hot DNA and place original bases in MLM labels."""

    masked_dna_sequence_one_hot = dna_sequence_one_hot.clone()
    valid = masked_dna_sequence_one_hot.sum(dim=-1).bool()
    mask = torch.rand(valid.shape, generator=generator) < 0.15
    mask &= valid
    labels = torch.full(valid.shape, -100, dtype=torch.long)
    labels[mask] = masked_dna_sequence_one_hot[mask].argmax(dim=-1)
    masked_dna_sequence_one_hot[mask] = 0.0
    batch_size = masked_dna_sequence_one_hot.shape[0]
    return DataBatch(
        dna_sequence_one_hot=masked_dna_sequence_one_hot.to(device),
        organism_index=torch.zeros(batch_size, dtype=torch.long, device=device),
        mlm=labels.to(device),
    )


def chromosome_length(fasta, chromosome: str, bigwig=None) -> int:
    """Return and validate the shared chromosome length."""

    try:
        fasta_length = fasta.get_reference_length(chromosome)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{chromosome!r} is absent from the FASTA.") from error

    if bigwig is None:
        return fasta_length
    bigwig_length = bigwig.chroms(chromosome)
    if bigwig_length is None:
        raise ValueError(f"{chromosome!r} is absent from the BigWig.")
    if bigwig_length != fasta_length:
        raise ValueError(
            f"Chromosome length differs between FASTA ({fasta_length}) and "
            f"BigWig ({bigwig_length})."
        )
    return fasta_length


def read_rna_signal(
    *,
    bigwig,
    chromosome: str,
    start: int,
    window_size: int,
) -> torch.Tensor:
    """Read one window of nonnegative 1-bp RNA-seq signal."""

    values = bigwig.values(
        chromosome,
        start,
        start + window_size,
        numpy=True,
    )
    values = np.nan_to_num(values, nan=0.0)
    if np.any(values < 0):
        raise ValueError("RNA-seq BigWig values must be nonnegative.")
    return torch.from_numpy(values.astype(np.float32, copy=False))


def rna_seq_batch(
    dna_sequence_one_hot: torch.Tensor,
    rna_signal: torch.Tensor,
    *,
    device: torch.device,
) -> DataBatch:
    """Convert collated DNA sequences and RNA-seq signal to a model batch."""

    batch_size = dna_sequence_one_hot.shape[0]
    return DataBatch(
        dna_sequence_one_hot=dna_sequence_one_hot.to(device),
        # NOTE: always zero (human in DeepMind JAX checkpoint)
        organism_index=torch.zeros(batch_size, dtype=torch.long, device=device),
        rna_seq=rna_signal.to(device),
    )


def calculate_nonzero_mean(
    dataset: RnaSeqWindowDataset,
    indices,
) -> float:
    """Return the exact nonzero mean without recounting window overlaps."""

    region_start = dataset.starts[indices[0]]
    region_end = dataset.starts[indices[-1]] + dataset.window_size
    total = 0.0
    count = 0
    for start in range(region_start, region_end, dataset.window_size):
        rna_signal = read_rna_signal(
            bigwig=dataset.bigwig,
            chromosome=dataset.chromosome,
            start=start,
            window_size=min(dataset.window_size, region_end - start),
        )
        nonzero = rna_signal[rna_signal != 0]
        total += nonzero.sum(dtype=torch.float64).item()
        count += nonzero.numel()
    if count == 0:
        raise RuntimeError("Training region contains no nonzero RNA-seq signal.")
    return total / count
