# Provenance: alphagenome-pt

# External
from __future__ import annotations
from collections.abc import Sequence
import torch

# Internal
from .schemas import DataBatch, OrganismIndex


_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _normalize_dna(
    sequences: str | Sequence[str] | None,
) -> list[str] | None:
    if isinstance(sequences, str):
        sequences = [sequences]
    elif sequences is not None:
        sequences = list(sequences)
    if sequences is not None:
        if not sequences or not all(isinstance(seq, str) for seq in sequences):
            raise TypeError("A DNA sequence batch must contain only strings.")
        sequence_lengths = {len(seq) for seq in sequences}
        if len(sequence_lengths) > 1:
            raise ValueError("DNA sequences in one batch must have equal lengths.")
    return sequences


def _normalize_dna_one_hot(
    one_hot: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if one_hot is not None:
        if one_hot.ndim == 2:
            one_hot = one_hot.unsqueeze(0)  # Add batch dimension
        if one_hot.ndim != 3 or one_hot.shape[-1] != 4:
            raise ValueError(
                "One-hot DNA must have shape [S, 4] or [B, S, 4], "
                f"got {tuple(one_hot.shape)}."
            )
        if one_hot.shape[0] == 0:
            raise ValueError("DNA sequence batches cannot be empty.")
        one_hot = one_hot.to(dtype=dtype)
    return one_hot


def _normalize_organism_index(
    batch: DataBatch,
    organism_index: OrganismIndex | None,
    num_organisms: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Normalize a passed organism index / the organism index of a passed batch.

    The batch may also be needed to infer batch size for normalization.

    Behavior:
    - Both are provided: Ensure they are equal and return one of them.
    - Only one is provided: Normalize and return it.
    - Neither is provided: Return a tensor of zeros with shape [B].
    """
    dna_one_hot = batch.dna_sequence_one_hot
    assert dna_one_hot is not None
    batch_size = dna_one_hot.shape[0]

    def normalize(index: OrganismIndex) -> torch.Tensor:
        index = torch.as_tensor(index)
        if index.dtype not in _INTEGER_DTYPES:
            raise TypeError("organism_index must contain integers.")
        if index.ndim == 0:
            index = index.repeat(batch_size)
        elif index.ndim == 2 and index.shape[-1] == 1:
            index = index.squeeze(-1)
        if index.shape != (batch_size,):
            raise ValueError(
                "organism_index must be a scalar or contain one value per "
                "batch item with shape [B] or [B, 1]; "
                f"got {tuple(index.shape)} for batch size {batch_size}."
            )
        index = index.to(device=device, dtype=torch.long)
        if torch.any((index < 0) | (index >= num_organisms)):
            raise ValueError(
                f"organism_index must be between 0 and {num_organisms - 1}."
            )
        return index

    batch_organism = (
        normalize(batch.organism_index)
        if batch.organism_index is not None
        else None
    )
    organism_index = (
        normalize(organism_index) if organism_index is not None else None
    )
    if batch_organism is not None and organism_index is not None:
        if not torch.equal(batch_organism, organism_index):
            raise ValueError(
                "If both organism indices are provided, they must be equal."
            )
        return batch_organism
    if batch_organism is not None:
        return batch_organism
    if organism_index is not None:
        return organism_index
    return torch.zeros(batch_size, dtype=torch.long, device=device)


def _normalize_splice_site_positions(
    splice_site_positions: torch.Tensor,
    batch: DataBatch,
) -> torch.Tensor:
    dna_onehot = batch.dna_sequence_one_hot
    assert dna_onehot is not None
    batch_size, seq_len, _ = dna_onehot.shape

    if splice_site_positions.dtype not in _INTEGER_DTYPES:
        raise TypeError("splice_site_positions must contain integer indices.")
    if (
        splice_site_positions.ndim != 3
        or splice_site_positions.shape[:2] != (batch_size, 4)
    ):
        raise ValueError(
            "splice_site_positions must have shape [B, 4, K], "
            f"got {tuple(splice_site_positions.shape)}."
        )

    splice_site_positions = splice_site_positions.to(
        device=dna_onehot.device,
        dtype=torch.int32,
    )
    invalid_position = (
        (splice_site_positions < -1) | (splice_site_positions >= seq_len)
    )
    if torch.any(invalid_position):
        raise ValueError(
            "splice_site_positions values must be -1 for padding "
            f"or between 0 and {seq_len - 1}."
        )
    return splice_site_positions
